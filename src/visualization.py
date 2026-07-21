"""Read-only entity visualization (Phase 10 patch 17) — a per-entity
timeline (swimlane) and a 1-hop relationship graph, both computed purely
from already-stored data (SQL reads only, no LLM calls anywhere in this
module). Kept free of any Streamlit/plotly/agraph import so this logic
stays testable independent of the GUI layer — app.py does the actual
chart/graph rendering and click-to-navigate wiring on top of these.
"""

from dataclasses import dataclass

from . import schema, storage

_EXCLUDED_FROM_GRAPH = {"timeline", "system"}

# Categorical palette (dataviz skill's validated reference instance, used
# unchanged — fixed hue order, never cycled). Only the first 7 of the
# validated 8 slots are used for real categories; slot 8 (red) is held
# back exclusively for the relationship graph's center node so a category
# color can never coincide with the "you are here" highlight. Categories
# beyond the 7th (schema grown past what this fixed set covers) fold to a
# neutral gray rather than reusing a slot.
_CATEGORY_PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7"]
_UNPLACED_CATEGORY_COLOR = "#898781"
CENTER_NODE_COLOR = "#e34948"

# Sequential blue ramp, ordinal-encoded (dataviz skill reference: steps
# >=250 so even the lightest step clears 2:1 contrast on a light surface).
# Used for edge color-by-weight, layered on top of the existing
# width-by-weight encoding.
_EDGE_WEIGHT_STEPS = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def filterable_categories() -> list:
    """Every schema category except timeline (that's the event itself, not
    a node) and system (world rules, never tied to specific events) —
    computed from the schema, never hardcoded, so a category added later
    via the dictionary GUI shows up as a filter checkbox automatically."""
    return [c for c in schema.list_categories() if c not in _EXCLUDED_FROM_GRAPH]


def get_participants(event: dict) -> list:
    """Every entity_id actually involved in a timeline event.

    A duration record's participants are exactly its entity/target pair
    (the same convention `creator._event_involved` and
    `rag_check`'s callers already use elsewhere in the codebase).

    A point record stores no participant list of its own — Phase 10's
    event-centric model keeps pointers only on the entity side (each
    involved entity's own event_ids references the event, not the other
    way around) — so the only way to answer "who's involved in this point
    event" is to check every entity's event_ids for this event's id.
    `location` is included naturally here (a location tagged into a point
    event gets a reciprocal event_ids pointer the same way any other
    involved entity does), which is intentional: a location where a key
    scene took place is a legitimate 1-hop relationship-graph neighbor."""
    if event.get("entity"):
        return [e for e in (event.get("entity"), event.get("target")) if e]
    participants = []
    for category in filterable_categories():
        for entity in storage.list_entities(category):
            if event["id"] in (entity.get("event_ids") or []):
                participants.append(entity["id"])
    return participants


def get_relationship_weight(entity_id: str, other_entity_id: str) -> int:
    """How many events entity_id and other_entity_id share — pure
    counting, no LLM. Straightforward reference implementation (each
    lookup rescans entity_id's own event list); `compute_neighbor_weights`
    below computes every neighbor's weight in one pass instead, for
    building a whole graph without rescanning per candidate."""
    events = storage.get_events_for_entity(entity_id)
    return sum(1 for e in events if other_entity_id in get_participants(e))


def _event_active_at(event: dict, year: int) -> bool:
    """Whether `event` counts as an existing connection as of `year` — a
    duration record (a relationship/status) only counts while active
    (start_year <= year <= end_year, or still ongoing if end_year is
    unset); a point record is a one-time fact that, once it's happened,
    stays true forever after (counts for any year >= its own), the same
    "no expiry once started" idea an open-ended duration already gets."""
    if event.get("entity"):
        start = event.get("start_year")
        if start is None or year < start:
            return False
        end = event.get("end_year")
        return end is None or year <= end
    point_year = event.get("year")
    return point_year is not None and point_year <= year


def compute_neighbor_weights(entity_id: str, as_of_year: int | None = None) -> dict:
    """{other_entity_id: shared_event_count} for every 1-hop neighbor of
    entity_id, in one pass over entity_id's own events (each event's
    participants are computed once, not once per candidate neighbor —
    same counts as calling get_relationship_weight for every neighbor,
    just without the redundant rescans).

    `as_of_year`, when given, drops any event not yet "existing" at that
    year (see `_event_active_at`) before counting it — the graph has no
    fixed "now" any more than the rest of this app does, so a relationship
    that's already ended, or a fact that hasn't happened yet as of the
    year being looked at, shouldn't still show up just because it exists
    somewhere on the full timeline."""
    weights: dict = {}
    for event in storage.get_events_for_entity(entity_id):
        if as_of_year is not None and not _event_active_at(event, as_of_year):
            continue
        for other_id in get_participants(event):
            if other_id == entity_id:
                continue
            weights[other_id] = weights.get(other_id, 0) + 1
    return weights


def category_color(category: str) -> str:
    """A stable hex per schema category. Sorted category order (not
    schema-file order, which could shuffle if entries are reordered) maps
    onto the fixed 7-slot categorical palette, so a given category keeps
    the same color across renders and across sessions as long as the
    category set itself doesn't change."""
    categories = sorted(filterable_categories())
    if category not in categories:
        return _UNPLACED_CATEGORY_COLOR
    index = categories.index(category)
    if index >= len(_CATEGORY_PALETTE):
        return _UNPLACED_CATEGORY_COLOR
    return _CATEGORY_PALETTE[index]


def edge_color(weight: int, min_weight: int, max_weight: int) -> str:
    """Sequential blue, light->dark — a higher shared-event count renders
    as a darker line, layered on top of the existing width-by-weight
    encoding. `min_weight`/`max_weight` scope the ramp to the range of
    weights actually shown (not weight's own theoretical range), so the
    color spread is meaningful for whatever's currently on screen."""
    if max_weight <= min_weight:
        return _EDGE_WEIGHT_STEPS[len(_EDGE_WEIGHT_STEPS) // 2]
    ratio = (weight - min_weight) / (max_weight - min_weight)
    index = round(ratio * (len(_EDGE_WEIGHT_STEPS) - 1))
    return _EDGE_WEIGHT_STEPS[index]


@dataclass
class GraphNode:
    entity_id: str
    label: str
    category: str


@dataclass
class GraphEdge:
    other_id: str
    weight: int


def build_relationship_graph(
    entity_id: str, weights: dict, allowed_categories: set, min_weight: int
) -> tuple:
    """Filter `weights` (from compute_neighbor_weights) down to the nodes/
    edges actually shown: category must be in `allowed_categories`, and
    edge weight must be >= min_weight. Returns (list[GraphNode],
    list[GraphEdge]) — plain data, no rendering."""
    nodes = []
    edges = []
    for other_id, weight in weights.items():
        if weight < min_weight:
            continue
        category = schema.category_from_id(other_id)
        if category is None or category not in allowed_categories:
            continue
        entity = storage.get_entity(category, other_id)
        label = (entity or {}).get("name") or other_id
        nodes.append(GraphNode(entity_id=other_id, label=label, category=category))
        edges.append(GraphEdge(other_id=other_id, weight=weight))
    return nodes, edges


@dataclass
class TimelineEntry:
    event_id: str
    kind: str  # "point" | "duration"
    label: str
    year: int | None = None  # point only
    start_year: int | None = None  # duration only
    end_year: int | None = None  # duration only (None = still ongoing)


def _target_label(target_id: str | None) -> str | None:
    if not target_id:
        return None
    category = schema.category_from_id(target_id)
    entity = storage.get_entity(category, target_id) if category else None
    return (entity or {}).get("name") or target_id


def build_timeline(entity_id: str) -> list:
    """This entity's own events (storage.get_events_for_entity is already
    year-sorted), reshaped into point/duration entries for the swimlane.
    Point labels are a plain truncation of `notes` — no LLM summarization,
    per spec. Duration labels are `predicate` plus the *other* party's name
    when one is set (e.g. "적대: 캐서린") — a relational record's entity/
    target pair isn't self-vs-other, it's just two sides, and entity_id can
    legitimately be sitting on either one (e.g. "미라가 데이비드를 알게 되었다" has
    entity=미라, target=데이비드; viewing 데이비드's own timeline must still label the
    *other* party (미라), not repeat back 데이비드's own id from the target field
    — caught by testing on the seed data's own `knows` record)."""
    entries = []
    for event in storage.get_events_for_entity(entity_id):
        if event.get("entity"):
            other_id = event.get("target") if event.get("entity") == entity_id else event.get("entity")
            label = event.get("predicate") or ""
            other_label = _target_label(other_id) if other_id and other_id != entity_id else None
            if other_label:
                label = f"{label}: {other_label}"
            entries.append(
                TimelineEntry(
                    event_id=event["id"], kind="duration", label=label,
                    start_year=event.get("start_year"), end_year=event.get("end_year"),
                )
            )
        else:
            notes = event.get("notes") or ""
            label = notes[:20] + ("..." if len(notes) > 20 else "")
            entries.append(
                TimelineEntry(event_id=event["id"], kind="point", label=label, year=event.get("year"))
            )
    return entries


# A duration event that's the chronologically last thing on record and
# still open needs *some* pixel of daylight past its own start_year to
# plot as a visible sliver rather than a zero-width stub — but that
# daylight is a rendering nudge, not a fact, so it must never leak into
# anything a human reads as "the year" (the cutoff's own label, or the
# relationship graph's year-slider bound): those read ReferenceLine.year
# (the real start_year, untouched), while only the plotted x-coordinate
# reads ReferenceLine.x (year + this buffer). A whole +2 years used to be
# applied to both alike, which is what made a duration event that started
# in (say) 2085 both plot AND *display* as if something dated "2087" had
# happened — this buffer only needs to be big enough for the bar's own
# `max(end - start, 0.5)` width floor (see _render_entity_timeline) to
# never have to stretch it further, so it stays this small on purpose.
_OPEN_DURATION_CUTOFF_BUFFER = 0.5


@dataclass
class ReferenceLine:
    year: int  # the real fact — always what gets displayed as "the year"
               # (a label, a slider bound, an axis annotation)
    x: float  # where this line/bound actually plots on the year axis —
              # equal to `year` for every real fact; only the duration-
              # flavored "cutoff" guess offsets it (see
              # _OPEN_DURATION_CUTOFF_BUFFER above)
    label: str  # the raw schema field name (birth_year, founded_year, ...)
                # or "마지막 이벤트" for a cutoff guess — no separate
                # i18n/friendly-name layer exists for field names anywhere
                # else in this app, so this stays consistent with that.
    is_guess: bool  # True only for the "cutoff" line — a stand-in, not a known fact


def resolve_timeline_reference(entity_id: str) -> dict:
    """Vertical reference lines for entity_id's own timeline, resolved
    once. Returns {"start": ReferenceLine|None, "end": ReferenceLine|None,
    "cutoff": ReferenceLine|None} — "end" and "cutoff" are mutually
    exclusive.

    "start"/"end" are entity_id's own lifecycle_start/lifecycle_end field
    values (birth_year/founded_year/created_year and death_year/
    destroyed_year/disbanded_year — whichever role its category defines),
    shown whenever set, for ANY category, not just character — a
    location's founding/destruction or an artifact's creation/destruction
    are exactly the same kind of fact. When "end" is set, it's also the
    bound an open-ended (end_year=None) duration bar should be capped
    at — nothing entity_id was doing continues past its own end.

    When lifecycle_start is NOT set, "start" falls back to the same kind
    of guess as "cutoff" below, just mirrored to the other end: entity_id's
    own chronologically FIRST recorded event's own start (a point event's
    year, or a duration event's start_year — never its end_year) —
    labeled "첫 기록" (first record) and marked is_guess=True, same as an
    unknown birth/founding year has to start being told from *somewhere*.

    "cutoff" only gets computed when "end" is absent, since nothing else
    would bound an open-ended bar in that case. It's based on whichever of
    entity_id's own events reaches furthest into the timeline — a point
    event's own year; a duration event's real end_year when set (a known
    fact, x == year); or, only for a duration event still genuinely open
    (no end_year), its start_year plus _OPEN_DURATION_CUTOFF_BUFFER purely
    so it still renders as a visible bar instead of a zero-length stub.
    Every event is compared this way and the single furthest one wins —
    NOT just get_events_for_entity's own last-sorted entry, which sorts a
    duration event by its start_year alone and so could pick an earlier-
    reaching event over one that started earlier but was later extended/
    closed much further out (caught via direct repro: an imprisoned status
    2000~2300 lost to an unrelated 2050 point event for "last" every
    time). is_guess=True regardless of which event or branch it came from.
    This replaces an earlier version that used the single latest year
    recorded anywhere in the WHOLE timeline table as a shared "now" — that
    looked fine until an entity's own newest event happened to equal that
    global value, which produced exactly the zero-length-stub bug this
    buffer exists to avoid (caught from a real screenshot, not
    anticipated). A later revision found the buffer itself creating a
    *different* confusion — a duration event that started in 2085 both
    plotting AND labeling as if "2087" were a real date — which is why
    the label/plot split above exists at all."""
    category = schema.category_from_id(entity_id)
    entity = storage.get_entity(category, entity_id) if category else None
    result = {"start": None, "end": None, "cutoff": None}
    if category and entity:
        start_fields = schema.get_fields_with_role(category, "lifecycle_start")
        if start_fields:
            value = entity.get(start_fields[0]["name"])
            if value is not None:
                result["start"] = ReferenceLine(year=value, x=value, label=start_fields[0]["name"], is_guess=False)
        end_fields = schema.get_fields_with_role(category, "lifecycle_end")
        if end_fields:
            value = entity.get(end_fields[0]["name"])
            if value is not None:
                result["end"] = ReferenceLine(year=value, x=value, label=end_fields[0]["name"], is_guess=False)

    if result["start"] is None or result["end"] is None:
        events = storage.get_events_for_entity(entity_id)
        if events:
            if result["start"] is None:
                first = events[0]
                # "Start" of the first event either way — a point event's
                # own year, or a duration event's start_year (never its
                # end_year, even if start_year itself is missing) — same
                # "no known fact, so guess from the earliest record" idea
                # as the cutoff below, just mirrored to the other end. No
                # buffer needed here — nothing renders as zero-width by
                # starting exactly where the chart's own left edge is.
                first_year = first.get("start_year") if first.get("entity") else first.get("year")
                if first_year is not None:
                    result["start"] = ReferenceLine(year=first_year, x=first_year, label="첫 기록", is_guess=True)
            if result["end"] is None:
                # Scanned across every event, not just events[-1] — that
                # was sorted by a duration event's own start_year (see
                # storage.get_events_for_entity), completely blind to its
                # end_year even when set. A status that started early but
                # was later extended/closed far in the future (e.g.
                # imprisoned 2000~2300) still sorted as if 2000 were its
                # only relevant year, so a later-STARTING point event
                # (say, 2050) wrongly won "last event" over a duration
                # record that actually reaches to 2300 — the cutoff line
                # then stayed stuck wherever it was even after the real
                # extent of the timeline had clearly moved (caught via a
                # direct repro, not anticipated up front). Each event's own
                # furthest known point — a duration's real end_year when
                # set, else its start_year plus the small guess buffer; a
                # point event's own year — is compared, and the single
                # furthest one wins, whichever event it came from.
                furthest_x, furthest_year = None, None
                for event in events:
                    if event.get("entity"):
                        end = event.get("end_year")
                        if end is not None:
                            x, year = end, end
                        else:
                            start = event.get("start_year") or 0
                            x, year = start + _OPEN_DURATION_CUTOFF_BUFFER, start
                    else:
                        x = year = event.get("year")
                    if x is None:
                        continue
                    if furthest_x is None or x > furthest_x:
                        furthest_x, furthest_year = x, year
                if furthest_x is not None:
                    result["cutoff"] = ReferenceLine(
                        year=furthest_year, x=furthest_x, label="마지막 이벤트", is_guess=True,
                    )
    return result
