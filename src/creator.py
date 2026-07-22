"""Creator (Phase 10 patch 22) — an autonomous reflection-loop feature that
drafts a multi-event narrative from a short user request, has Inspector
(the existing Step 4/5 validation pipeline) check each drafted event, and
retries on rejection until it passes or a retry cap is hit.

Unlike the normal chat pipeline, Creator never calls inference.infer_event —
its own narrative-composition logic (event count, point/duration, entity/
target, predicate) replaces Step 3's job entirely for this flow. Each
drafted event still gets real Korean prose (used as both the timeline
record's notes and the raw_text fed into rag_check's checks), so Step 4/5
run completely unmodified.
"""

import json
import re
from dataclasses import dataclass, field

from . import archivist, config, hard_check, rag_check, schema, storage

MAX_EVENTS = 5
MAX_NEW_ENTITIES = 3

# Categories Creator can ever touch (new-entity creation or backdrop
# reference) are every schema category except system (world rules, never
# author-narrated) and timeline (that's the event itself, not a cast/prop
# entity). Computed from the schema, not a hardcoded list, so a category
# added later shows up automatically — same reasoning as the GUI checkboxes.
_EXCLUDED_CATEGORIES = {"system", "timeline"}
# Categories whose *existing* entities Creator may reference naturally even
# when not tagged — backdrop, not narrative agents (a sword or a tavern
# doesn't have agency; a character does). character/race stay tagged-only
# for existing entities regardless of the new-entity-creation toggles below
# — if the user wants an existing character involved, they tag them; the
# risk of an uninvited character with their own history showing up in a
# story isn't worth the convenience, unlike an uninvited sword or building.
_BACKDROP_CATEGORIES = {"location", "artifact", "faction"}


def eligible_categories() -> list:
    return [c for c in schema.list_categories() if c not in _EXCLUDED_CATEGORIES]

_TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_YEAR_RANGE_PATTERN = re.compile(r"(\d+)\s*년?\s*(?:부터|[~\-])\s*(\d+)\s*년(?:\s*까지)?")
# "2090년과 2100년 사이" / "2090년와 2100년 사이" — a separate pattern from
# _YEAR_RANGE_PATTERN because "과/와 ... 사이" doesn't share a common
# separator token with "부터...까지"/"~"/"-". Missing this meant a request
# stating an explicit "A년과 B년 사이" range fell through to the no-hint
# branch entirely — the year-confirm screen then showed the unrelated
# auto-computed existence window (e.g. entities' full 2000-onward
# coexistence range) instead of the range the user actually typed, even
# though the request never intended to ask for a confirmation at all.
_YEAR_BETWEEN_PATTERN = re.compile(r"(\d+)\s*년\s*(?:과|와)\s*(\d+)\s*년\s*사이")
_YEAR_PATTERN = re.compile(r"(\d+)\s*년")


def parse_year_hint(text: str) -> tuple:
    """(lower, upper) if `text` explicitly states a year or year range
    ("2010년", "2000~2010년", "2000년과 2010년 사이"), else (None, None)
    meaning no explicit year was given at all. A bare single year returns
    (y, y) — the caller decides what a single value means (compose_narrative's
    is_single_year branch handles the actual constraint; creator_session
    decides whether to run the count-mismatch check). Bracket contents are
    excluded from the scan, same reasoning as parser.parse_input: a tag like
    "[100년 전쟁]" isn't a year mention. Two or more loose year mentions with
    no explicit range separator are ambiguous and treated as no hint at all,
    falling through to the auto-computed-window flow."""
    stripped = _TAG_PATTERN.sub(" ", text)
    range_match = _YEAR_RANGE_PATTERN.search(stripped) or _YEAR_BETWEEN_PATTERN.search(stripped)
    if range_match:
        a, b = int(range_match.group(1)), int(range_match.group(2))
        return (a, b) if a <= b else (b, a)
    years = sorted({int(m) for m in _YEAR_PATTERN.findall(stripped)})
    if len(years) == 1:
        return years[0], years[0]
    return None, None


@dataclass
class YearWindow:
    lower: int | None  # None = unbounded below
    upper: int | None  # None = unbounded above (still exists/ongoing)
    possible: bool
    reason: str | None = None  # set only when not possible
    per_entity: dict = field(default_factory=dict)  # entity_id -> (lower, upper)


def _pairwise_connection_floor(entity_ids: list) -> tuple:
    """Earliest start_year among duration records connecting any two of the
    given entities to *each other* — e.g. a "knows"/"enemies_with" record
    between them is proof they'd had at least one point of contact by then,
    which floors any new joint scene involving both. Without this, the
    suggested window only looked at each entity's own existence range and
    could offer years technically within both characters' lifespans but
    still before they'd ever met (observed in practice: two characters on
    record as first meeting in 2079 still got offered a range starting in
    the 2010s-2030s, purely from birth years, and every attempt in that
    span failed Inspector). Returns (year, reason) or (None, None) if no
    such cross-entity connection exists on record."""
    entity_id_set = set(entity_ids)
    earliest = None
    earliest_pair = None
    for entity_id in entity_ids:
        for record in storage.get_duration_records(entity_id):
            other = record.get("target") if record.get("entity") == entity_id else record.get("entity")
            if other not in entity_id_set or other == entity_id:
                continue
            start = record.get("start_year")
            if start is not None and (earliest is None or start < earliest):
                earliest = start
                earliest_pair = (entity_id, other, record.get("predicate"))
    if earliest is None:
        return None, None
    a, b, predicate = earliest_pair
    reason = f"{a}와(과) {b}의 관계('{predicate}')가 {earliest}년부터 시작된 것으로 기록되어 있습니다."
    return earliest, reason


def compute_year_window(entity_ids: list) -> YearWindow:
    """Intersects every entity's own existence range (hard_check.
    get_existence_range) into one window a Creator-generated story must fit
    within, then further raises the lower bound to account for any
    recorded relationship *between* the given entities (see
    _pairwise_connection_floor). `possible=False` means the resulting
    window is empty — the caller should reject the request before ever
    invoking Creator, not burn a retry loop on something that can never
    pass Inspector."""
    per_entity = {}
    for entity_id in entity_ids:
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        per_entity[entity_id] = hard_check.get_existence_range(category, entity_id)

    lower = None
    upper = None
    for e_lower, e_upper in per_entity.values():
        if e_lower is not None:
            lower = e_lower if lower is None else max(lower, e_lower)
        if e_upper is not None:
            upper = e_upper if upper is None else min(upper, e_upper)

    if lower is not None and upper is not None and lower > upper:
        ended_before = [eid for eid, (_, u) in per_entity.items() if u is not None and u < lower]
        started_after = [eid for eid, (l, _) in per_entity.items() if l is not None and l > upper]
        reason = (
            f"{', '.join(ended_before)}의 존재가 끝난 시점({upper}년)이 "
            f"{', '.join(started_after)}의 존재가 시작된 시점({lower}년)보다 이릅니다 — "
            f"함께 존재하는 기간이 없습니다."
        )
        return YearWindow(lower=lower, upper=upper, possible=False, reason=reason, per_entity=per_entity)

    connection_floor, connection_reason = _pairwise_connection_floor(entity_ids)
    if connection_floor is not None and (lower is None or connection_floor > lower):
        if upper is not None and connection_floor > upper:
            return YearWindow(
                lower=connection_floor, upper=upper, possible=False,
                reason=f"{connection_reason} 하지만 관련 엔티티의 존재 기간은 {upper}년까지입니다.",
                per_entity=per_entity,
            )
        lower = connection_floor

    return YearWindow(lower=lower, upper=upper, possible=True, per_entity=per_entity)


# ---------------------------------------------------------------------------
# Narrative composition — Creator's own replacement for Step 3 in this flow
# ---------------------------------------------------------------------------

@dataclass
class DraftEvent:
    event_type: str  # "point" | "duration"
    notes: str
    involved_entities: list = field(default_factory=list)
    year: int | None = None  # point events only
    location: str | None = None  # point events only — an existing location's entity_id
    # duration events: start_year/end_year live inside duration_effect
    # (mirrors inference.InferredEvent's exact convention), not as separate
    # top-level fields — archivist._build_duration_diff reads them from
    # duration_effect directly, so this isn't just cosmetic consistency.
    duration_effect: dict | None = None

    @property
    def start_year(self) -> int | None:
        return (self.duration_effect or {}).get("start_year")

    @property
    def end_year(self) -> int | None:
        return (self.duration_effect or {}).get("end_year")


@dataclass
class DraftEntity:
    """A brand-new supporting entity Creator invented (only ever possible
    when its category's checkbox is on). `entity_id` is already a real,
    final id (minted via archivist.generate_id the moment the draft is
    parsed, same collision-checked mechanism timeline records already use)
    — not a placeholder needing later resolution, so every DraftEvent field
    that references it is a normal entity_id string from the start, same
    as any pre-existing entity. Nothing is written to storage until the
    draft is human-approved and saved; minting the id early is just a
    string computation (archivist.generate_id never writes), so a
    discarded/retried draft leaves nothing behind to clean up."""

    entity_id: str
    category: str
    fields: dict  # e.g. {"name": "밥", "notes": "...", "category": "용병단"}


@dataclass
class NarrativeDraft:
    events: list  # list[DraftEvent]
    # Creator's own unconstrained judgment of how many events this story
    # ideally wants — reported honestly even when `events` itself had to
    # comply with a single-year constraint (verified: a 4-event story
    # compressed into one year still reports natural_event_count=4, not a
    # constraint-distorted 1), so the caller can detect a count/year-shape
    # mismatch (spec section B) purely as `is_single_year and
    # natural_event_count > 1` without a wasted extra LLM round-trip when
    # the user picks "compress" rather than "widen the range". An earlier
    # version also asked the LLM to self-report a `would_prefer_range`
    # boolean directly, but that came back inconsistent (False even for
    # the same 4-into-1-year case) — dropped in favor of this simpler,
    # structural signal that doesn't depend on a second subjective judgment.
    natural_event_count: int
    new_entities: list = field(default_factory=list)  # list[DraftEntity]


def _get_llm():
    # temperature=0.7, unlike every other LLM call in this codebase
    # (always 0) — those are classification/judgment calls that need to be
    # reproducible; this one is creative drafting, and a retry after
    # Inspector rejection should actually explore a different narrative,
    # not deterministically regenerate the same rejected draft.
    return config.get_chat_model("reasoning", temperature=0.7)


def _invoke_llm(prompt: str) -> str:
    response = _get_llm().invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾을 수 없습니다: {raw!r}")
    return json.loads(match.group(0))


def _invoke_llm_json(prompt: str, attempts: int = 2) -> dict:
    """_invoke_llm + _extract_json, retried up to `attempts` times on a
    parse failure — same reasoning and shape as rag_check._invoke_llm_json:
    a malformed/truncated response used to raise straight through as an
    uncaught exception instead of just being retried."""
    last_error = None
    for _ in range(attempts):
        raw = _invoke_llm(prompt)
        try:
            return _extract_json(raw)
        except ValueError as exc:
            last_error = exc
    raise last_error


def _range_overlaps(e_lower, e_upper, lower: int, upper: int) -> bool:
    """False only when the two ranges are provably disjoint — None on
    either side of the entity's own range means unbounded in that
    direction, so it can never be the side that disqualifies it."""
    if e_upper is not None and e_upper < lower:
        return False
    if e_lower is not None and e_lower > upper:
        return False
    return True


def _backdrop_entity_ids(lower: int, upper: int) -> dict:
    """{entity_id: (label, category)} for every backdrop-category
    (location/artifact/faction) entity whose own existence range
    (hard_check.get_existence_range — founded/created_year ~
    destroyed_year) overlaps [lower, upper] — the actual set of ids
    Creator is shown as freely referenceable without tagging (see
    _backdrop_entities_block, built from this same set). Shared with
    compose_narrative's own reference validation: these entities are
    legitimately usable without ever being tagged in the current request
    (e.g. reusing loc_은빛도시_지하감옥 from an already-existing imprisonment
    record tagged only in some earlier, separate request) — validating
    against just resolved_entities/tag_to_id alone rejected exactly this
    as if it were a hallucinated id, when it was a perfectly real,
    intentionally-always-usable one (caught via direct repro, not
    anticipated when the validation was first added).

    Filtered by existence range for the same reason
    _backdrop_entities_block always was: a destroyed-in-1200 location
    still showing up as an option for a 2055-2059 story meant Creator kept
    picking it, hard_check kept rejecting it, and every one of 4 retries
    burned itself out on the same unwinnable choice. Filtering here means
    it's never offered in the first place, the same "fix it before
    Creator runs, not after" principle the year-window computation already
    applies to characters."""
    result = {}
    for category in sorted(_BACKDROP_CATEGORIES & set(schema.list_categories())):
        for e in storage.list_entities(category):
            e_lower, e_upper = hard_check.get_existence_range(category, e["id"])
            if not _range_overlaps(e_lower, e_upper, lower, upper):
                continue
            result[e["id"]] = (e.get("name") or e["id"], category)
    return result


def _backdrop_entities_block(backdrop_ids: dict) -> str:
    """Existing entities of the backdrop categories (location/artifact/
    faction) — shown so Creator can naturally reference them (Phase 10
    patch 22, B), same reasoning as the original location-only version this
    replaces: a duel referencing an existing named sword, or a scene set at
    an existing tavern, enriches a story without the narrative-agency risk
    an uninvited *character* would carry (see module-level comment on
    _BACKDROP_CATEGORIES)."""
    lines = [
        f'- {entity_id} ("{label}", {category})'
        for entity_id, (label, category) in backdrop_ids.items()
    ]
    return "\n".join(lines) if lines else "(등록된 항목 없음)"


def _new_entity_category_block(allowed_new_categories: set) -> str:
    lines = []
    for category in sorted(allowed_new_categories):
        required = schema.get_required_fields(category)
        field_descs = []
        for f in required:
            if f["type"] == "enum":
                field_descs.append(f"{f['name']} (필수, 다음 중 하나: {', '.join(f.get('options') or [])})")
            else:
                field_descs.append(f"{f['name']} (필수)")
        lines.append(f"- {category}: " + (", ".join(field_descs) if field_descs else "(필수 필드 없음)"))
    return "\n".join(lines)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def _matching_tagged_entity(category: str, name: str, resolved_entities: dict) -> str | None:
    """The real entity_id already tagged for this request whose own stored
    name is the same as, or a 1-character hallucinated slip away from,
    `name` — or None if nothing in resolved_entities is a plausible match.

    Caught in practice: Creator, mid-draft, invented a brand-new "new_entities"
    character named "쟩" — a single-final-consonant slip of "데이비드" (ㅇ vs ㄴ),
    the SAME character the user had already tagged and resolved in this
    exact request — creating a redundant near-duplicate entity instead of
    just referencing the real one. Checking only against entities already
    tagged for THIS request (not the whole world's entities of that
    category) keeps this narrow: Creator has no legitimate reason to
    "invent" a new character this similar to someone the user just
    explicitly named a moment ago, so a match here is treated as the same
    person, not two coincidentally similar strangers."""
    for entity_id in resolved_entities.values():
        if schema.category_from_id(entity_id) != category:
            continue
        record = storage.get_entity(category, entity_id)
        existing_name = (record or {}).get("name")
        if not existing_name:
            continue
        if existing_name == name or _levenshtein(existing_name, name) <= 1:
            return entity_id
    return None


def _closest_valid_id(bad_id: str, valid_ids: set) -> str | None:
    """The single valid id within edit-distance 1 of `bad_id`, if any —
    the same "쟩 for the real 데이비드" tolerance _matching_tagged_entity applies
    to a proposed new_entities name, generalized to any raw entity_id a
    drafted event carries directly (duration_effect.entity/target,
    involved_entities, location). Nothing stops that exact same
    hallucination from showing up as a bare reference instead of a
    new_entities proposal — caught in practice: a retry that no longer
    invented a duplicate character still separately hallucinated
    "char_쟩" straight into a duration_effect, which the reference
    validation correctly rejected as unrecognized, but rejecting outright
    (forcing a full recompose) is worse than just fixing an obvious
    one-character slip in place."""
    for candidate in valid_ids:
        if _levenshtein(bad_id, candidate) <= 1:
            return candidate
    return None


def _resolve_open_target(entity_id: str | None, predicate: str | None) -> str | None:
    """The target of whatever (entity_id, predicate) record is currently
    open in storage (end_year is None), if there's exactly one. Used to
    fill in a clear/set_closed duration_effect's target when the LLM left
    it null — only safe when unambiguous, so 0 or 2+ open matches return
    None rather than guessing."""
    if not entity_id or not predicate:
        return None
    open_targets = {
        event.get("target")
        for event in storage.get_events_for_entity(entity_id)
        if event.get("entity") == entity_id
        and event.get("predicate") == predicate
        and event.get("end_year") is None
    }
    if len(open_targets) == 1:
        return next(iter(open_targets))
    return None


def _resolve_new_entities(data: dict, allowed_new_categories: set, resolved_entities: dict) -> tuple:
    """Parses data["new_entities"], mints each a real entity_id immediately
    (archivist.generate_id — collision-checked against storage AND every id
    minted earlier in this same call, exactly how timeline ids within one
    batch already avoid colliding), and returns (list[DraftEntity], {tag:
    real_id}) so callers can rewrite every event's entity references from
    Creator's own placeholder tags to real ids in one pass. Silently drops
    any entity whose category isn't in allowed_new_categories or is missing
    a name — Creator ignoring its own instructions shouldn't crash the
    draft, just lose that one entity.

    Before minting anything, checks whether the proposed name is really
    just a hallucinated near-duplicate of an entity already tagged this
    request (see _matching_tagged_entity) — if so, the tag is remapped
    straight to that real, existing entity_id instead of creating a
    phantom duplicate character."""
    entities = []
    tag_to_id = {}
    existing_ids = set()
    for raw in (data.get("new_entities") or [])[:MAX_NEW_ENTITIES]:
        tag = raw.get("tag")
        category = raw.get("category")
        fields = dict(raw.get("fields") or {})
        if not tag or category not in allowed_new_categories or not fields.get("name"):
            continue
        duplicate_of = _matching_tagged_entity(category, fields["name"], resolved_entities)
        if duplicate_of:
            tag_to_id[tag] = duplicate_of
            continue
        entity_id = archivist.generate_id(category, fields["name"], existing_ids)
        existing_ids.add(entity_id)
        tag_to_id[tag] = entity_id
        entities.append(DraftEntity(entity_id=entity_id, category=category, fields=fields))
    return entities, tag_to_id


def _autopromote_undeclared_ids(
    data: dict, allowed_new_categories: set, tag_to_id: dict, new_entities: list
) -> None:
    """Mutates tag_to_id/new_entities in place to cover a gap distinct from
    _closest_valid_id's typo tolerance: the LLM sometimes invents a brand-new
    entity id directly inside an event's location/involved_entities/
    duration_effect fields (following the id_prefix convention it sees in
    the existing-entity list, e.g. "loc_바람길_황무지") instead of declaring it
    through the new_entities array the prompt asks for. That id isn't a
    near-miss of any real id, so _autocorrect can't save it, and the whole
    request failed outright even with the category's checkbox explicitly
    checked (observed directly: 'location' checked, LLM still wrote a bare,
    undeclared location id straight into event.location).

    Only promotes an id whose schema.category_from_id lands in
    allowed_new_categories (what the user actually opted into), that isn't
    already a real stored entity (a real id merely filtered out of this
    request's backdrop list — e.g. by the existence-window filter — should
    still fail as an out-of-scope reference, not get silently duplicated
    under a new id), and only up to the same MAX_NEW_ENTITIES budget
    _resolve_new_entities enforces, so this can't smuggle in an unbounded
    number of invented entities."""
    existing_ids = set(tag_to_id.values())
    seen_raw = set()
    for e in data.get("events") or []:
        duration_effect = e.get("duration_effect") or {}
        refs = list(e.get("involved_entities") or [])
        refs.append(e.get("location"))
        refs.append(duration_effect.get("entity"))
        refs.append(duration_effect.get("target"))
        for ref in refs:
            if not ref or not isinstance(ref, str) or ref in tag_to_id or ref in seen_raw:
                continue
            category = schema.category_from_id(ref)
            if category not in allowed_new_categories:
                continue
            if storage.get_entity(category, ref) is not None:
                continue
            if len(new_entities) >= MAX_NEW_ENTITIES:
                return
            seen_raw.add(ref)
            prefix = schema.load_schema_registry()[category]["id_prefix"]
            name = ref[len(prefix):].replace("_", " ").strip() or ref
            entity_id = archivist.generate_id(category, name, existing_ids)
            existing_ids.add(entity_id)
            tag_to_id[ref] = entity_id
            new_entities.append(DraftEntity(entity_id=entity_id, category=category, fields={"name": name}))


def _remap_tags(value, tag_to_id: dict):
    if isinstance(value, list):
        return [tag_to_id.get(v, v) for v in value]
    if isinstance(value, str):
        return tag_to_id.get(value, value)
    return value


def _entity_context_block(resolved_entities: dict) -> str:
    """Tagged entities' own stored fields/notes, PLUS each one's related
    duration/point records (ownership, membership, past events) — found
    missing in practice: this used to show only the entity's own fields and
    notes, never its duration history, so Creator had no way to know a
    tagged character already owns a specific named artifact (a duration
    'owns' record) and kept falling back to generic language ("검", "명검")
    even when the character's own Excalibur-owning record was sitting in
    storage the whole time. Inspector's own context
    (rag_check._entity_context_lines) already included this; Creator's
    drafting context just hadn't been given the same information.

    Deduped by event id (token-diet pass): two tagged entities that share
    most of their history (e.g. two characters who adventure together)
    used to each print every shared event's notes in full under their own
    section — measured at 40% duplicate lines for a real pair with ~15
    shared events, and that duplication is resent in full on every single
    retry. Each real event is now printed exactly once, attributed to
    every tagged entity it touches on one combined line."""
    lines = []
    seen_events: dict = {}
    event_order = []
    for tag, entity_id in resolved_entities.items():
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        record = storage.get_entity(category, entity_id)
        if not record:
            continue
        parts = [f"분류={category}"]
        summary = rag_check.entity_field_summary(record)
        if summary:
            parts.append(summary)
        if record.get("notes"):
            parts.append(f"notes={record['notes']}")
        lines.append(f'{entity_id} ("{tag}"): ' + ", ".join(parts))
        for related_event in storage.get_events_for_entity(entity_id):
            range_note = rag_check._duration_range_note(related_event)
            if not related_event.get("notes") and not range_note:
                continue
            eid = related_event["id"]
            if eid not in seen_events:
                seen_events[eid] = {"entities": [], "event": related_event, "range_note": range_note}
                event_order.append(eid)
            if entity_id not in seen_events[eid]["entities"]:
                seen_events[eid]["entities"].append(entity_id)

    for eid in event_order:
        info = seen_events[eid]
        entity_label = ", ".join(info["entities"])
        line = f"{entity_label}의 관련 기록({eid}): {info['event'].get('notes') or ''}"
        if info["range_note"]:
            # The real, already-stored start_year/end_year — spelled out
            # explicitly regardless of what notes says, since notes is
            # free prose that never gets rewritten when a field like
            # end_year is edited independently afterward (e.g. via the GUI
            # field editor). Without this, Creator has no way to know a
            # status it's about to narrate the end of already has a real,
            # fixed end date on file, and just invents its own instead
            # (caught in practice: a fresh "released in 2086" event
            # drafted for a status whose real end_year was already 2090).
            line += f" {info['range_note']}"
        lines.append(line)
    return "\n".join(f"- {line}" for line in lines) if lines else "(참고할 엔티티 정보 없음)"


def compose_narrative(
    resolved_entities: dict,
    request_text: str,
    lower: int,
    upper: int,
    feedback: str | None = None,
    supplement: str | None = None,
    allowed_new_categories: set | None = None,
) -> NarrativeDraft:
    """Draft a multi-event narrative for `request_text`, entirely within
    [lower, upper] (inclusive, both concrete ints — the caller resolves any
    open-ended YearWindow into concrete bounds before calling this, e.g.
    via user confirmation). Bypasses inference.infer_event entirely — this
    function decides event_type/duration_effect/predicate/target itself,
    replacing Step 3's job for this flow (see module docstring); Step 4/5
    validation (Inspector) still runs against the notes text this produces,
    completely unmodified.

    `feedback`, when given, is Inspector's rejection reason(s) from a prior
    failed attempt in the same retry loop. `supplement` is an optional
    user-provided instruction from a [Redo] request — added on top of the
    original request, never replacing it. `allowed_new_categories` (Phase 10
    patch 22, B — off/empty by default) is the set of categories the user's
    per-category checkboxes allow Creator to *invent new* entities in;
    existing entities of the backdrop categories (location/artifact/
    faction) are always referenceable regardless of this set — it only
    gates fabricating brand-new ones."""
    allowed_new_categories = allowed_new_categories or set()
    entity_list = "\n".join(f'- "{tag}" -> {entity_id}' for tag, entity_id in resolved_entities.items())
    entity_context = _entity_context_block(resolved_entities)
    valid_ids = ", ".join(resolved_entities.values())

    # Existing location/artifact/faction entities (Phase 10 patch 22, B) —
    # Creator can naturally reference any of these in a scene ("은빛도시에서
    # 얘기를 나누었다", "여명검을 휘둘렀다"), same as a person narrating a
    # scene would, without needing them tagged. character/race are
    # deliberately excluded here — see _BACKDROP_CATEGORIES.
    backdrop_ids = _backdrop_entity_ids(lower, upper)
    backdrop_block = _backdrop_entities_block(backdrop_ids)

    new_entity_block = ""
    if allowed_new_categories:
        category_block = _new_entity_category_block(allowed_new_categories)
        new_entity_block = (
            "\n\n=== 새로운 조연 엔티티 생성 ===\n"
            f"다음 카테고리는 필요하다면 새로 지어내도 좋다 (최대 {MAX_NEW_ENTITIES}개까지, "
            "이야기에 실제로 필요한 만큼만 — 억지로 채우지 마라):\n"
            f"{category_block}\n"
            "각 새 엔티티는 new_entities 배열에 {\"tag\": 이 응답 안에서만 쓰는 임시 식별자, "
            "\"category\": 카테고리, \"fields\": {필수 필드 전부 + 선택적으로 notes 등}}로 "
            "채워라. events 안에서 그 엔티티를 참조할 때는(involved_entities, "
            "duration_effect.entity/target, location) 실제 entity_id 대신 이 tag 문자열을 "
            "그대로 써라 — 실제 id는 이후 자동으로 부여된다. 이름 없는 '여러 사람', '누군가' "
            "같은 뭉뚱그린 표현 대신, 서사에 필요하다면 구체적인 조연으로 만들어라(예: '카라반 "
            "마스터 밥'). 이 목록에 없는 카테고리로는 절대 새 엔티티를 만들지 마라."
        )
        allowed_note = (
            f" (단, {', '.join(sorted(allowed_new_categories))}은(는) 아래 안내에 따라 새로 "
            "지어내도 좋다)"
        )
    else:
        allowed_note = ""

    all_status_effects = schema.load_status_effects()

    def _effect_line(s: dict) -> str:
        line = f"- {s['id']} ({s['label']})"
        if s.get("notes"):
            line += f": {s['notes']}"
        return line

    status_effect_options = "\n".join(
        _effect_line(s) for s in all_status_effects if s.get("type", "individual") == "individual"
    ) or "(등록된 개인 상태 predicate 없음)"
    relational_predicate_options = "\n".join(
        _effect_line(s) for s in all_status_effects if s.get("type") == "relational"
    ) or "(등록된 관계형 predicate 없음)"

    is_single_year = lower == upper
    year_constraint = (
        f"{lower}년 (단일 연도 — 모든 사건은 반드시 이 연도 하나로만 채워야 한다)"
        if is_single_year
        else f"{lower}년 ~ {upper}년 (이 범위 밖의 연도는 절대 쓰지 마라)"
    )

    single_year_instruction = (
        "\n\n이 요청은 단일 연도 하나로 제한되어 있다. natural_event_count는 이 제약과 "
        "무관하게, '만약 제약이 없었다면 몇 개의 사건으로 구성하는 게 이상적이었을지'를 "
        "정직하게 보고하라 — 실제로 events를 몇 개 작성했는지에 맞춰 축소해서 보고하지 마라 "
        "(예: 이상적으로는 4개가 자연스러운 이야기라면, 단일 연도 제약 때문에 실제로는 다르게 "
        "압축해서 작성하더라도 natural_event_count는 여전히 4여야 한다). events 자체는 "
        "그럼에도 불구하고 반드시 위 단일 연도 제약을 지켜서 최대한 압축된 형태로 작성하라."
        if is_single_year
        else ""
    )
    feedback_block = (
        f"\n\n이전 시도가 다음 이유로 반려되었다 — 이번에는 이 문제를 피해서 다시 구성하라:\n{feedback}"
        if feedback
        else ""
    )
    supplement_block = (
        f"\n\n사용자가 재생성 시 추가로 요청한 지침(원래 요청에 덧붙여 반영하라): {supplement}"
        if supplement
        else ""
    )

    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 이야기 기획자(Creator)다. 사용자의 짧은 요청을 "
        "받아, 그 요청을 표현하는 하나 이상의 사건 기록(timeline record) 초안을 스스로 작성하라. "
        f"아래 확정된 엔티티만 사용하고, 새로운 인물/종족은 절대 지어내지 마라{allowed_note}.\n\n"
        f"확정된 엔티티:\n{entity_list}\n\n"
        f"엔티티 정보:\n{entity_context}\n\n"
        f"사용 가능한 entity_id: {valid_ids}\n\n"
        f"사용자 요청: {request_text}\n\n"
        f"허용된 연도: {year_constraint}\n\n"
        "=== 사건 개수/구성 판단 ===\n"
        "이 서사가 몇 개의 사건 기록으로 표현되는 게 자연스러운지 스스로 판단하라 — 고정된 "
        f"개수나 하한은 없다(응집도 높은 단일 사건이면 1개로 충분하다), 상한은 {MAX_EVENTS}개다. "
        "예: '원수가 됐다'는 보통 응집도 높은 단일 사건으로 충분하다. '사랑에 빠지기까지'처럼 "
        "과정 자체가 여러 단계(만남, 데이트, 고백 등)로 구성되는 게 자연스러운 서사는 여러 개의 "
        "point 사건으로 나누는 게 좋다.\n\n"
        "=== duration 이벤트 포함 여부 ===\n"
        "point 사건들의 결과가 실제로 지속되는 상태/관계의 성립으로 자연스럽게 귀결되는 경우에만 "
        "마지막에 duration 이벤트를 추가하라 — 여러 사건을 만든다고 항상 duration도 만들어야 "
        "하는 건 아니다. 예: 사랑 이야기라면 마지막에 연인 관계 duration을 추가하는 게 자연스럽지만, "
        "단순히 바보짓을 하는 이야기라면 point만으로 완결되고 duration은 불필요하다.\n\n"
        "위 '엔티티의 관련 기록'에서 실제 기간이 이미 종료된 것으로 표시된(예: \"실제 기간: "
        "2085년~2090년, 이미 종료됨\") 상태/관계는, 그 기록 자체로 이미 완결된 확정 사실이다 — "
        "그 종료를 다시 서술하는 새 이벤트(point든 duration이든)를 만들 필요가 전혀 없다. 그 "
        "상태가 그 종료 연도 이후로는 이미 끝나 있다는 것을 그냥 전제로 삼고, 곧바로 그 다음 "
        "이야기(사용자가 실제로 요청한 내용)를 써라 — 이미 끝난 일을 다시 '풀려났다', "
        "'해제됐다' 같은 문장으로 새로 만들어 사건 하나를 낭비하지 마라.\n\n"
        "duration_effect.action은 다음 중 하나다 — 특히 clear는 이미 열려 있는 기존 상태/관계를 "
        "실제로 종료시키는 유일한 방법이다: 새로 별개의 duration 레코드를 만들어 '해제되었다'고 "
        "서술하는 것만으로는 기존 기록이 실제로 닫히지 않는다 (그 기존 레코드의 end_year는 "
        "그대로 비어있는 채 남는다). 위 '엔티티의 관련 기록'에 이미 열려 있는(end_year 없는) "
        "상태/관계가 있고, 이번 서사가 그것을 끝내는 내용이라면 반드시 clear를 써서 그 기존 "
        "기록 자체를 닫아라 (이미 종료된 기록은 위 문단대로 그대로 두고 새로 clear하지 마라):\n"
        "  - set: 새로운 상태/관계가 이 사건에서 시작됨. start_year 필요, end_year는 없음.\n"
        "  - clear: 이미 열려 있는 기존 상태/관계가 이 사건에서 끝남 (예: 추방이 풀렸다, 수감에서 "
        "석방됐다, 단체가 해체됐다). entity와 predicate는 반드시 그 기존 열린 기록과 정확히 "
        "동일해야 그 기록을 찾아 닫을 수 있다. end_year(이 사건의 연도) 필수, start_year는 "
        "없어도 된다.\n"
        "  - set_closed: 이미 시작과 끝이 모두 지난 상태/관계를 한 번에 서술함 (예: '2050년부터 "
        "2060년까지 수감되어 있었다'). start_year와 end_year 둘 다 필요.\n\n"
        "관계형 predicate에서, 이번 서사가 이미 열려 있는 관계와 양립할 수 없는 정반대의 새 "
        "관계를 성립시킨다면(예: 원수 관계였던 두 사람이 화해하여 친구가 되는 경우, 동맹이었던 "
        "세력이 배신하여 적대 관계가 되는 경우) — 기존 관계 옆에 새 관계를 별개로 set하지 "
        "마라. 반드시 먼저 기존의 상충되는 관계를 그 predicate 그대로 clear로 닫고, 그 다음에 "
        "새로운 관계를 별도 사건으로 set하라. 화해/배신/절교/결별처럼 관계가 뒤바뀌는 서술은 "
        "새 관계의 시작이자 동시에 기존 관계의 종료를 의미하며, 종료를 명시하지 않으면 두 "
        "관계가 동시에 열린 채로 남아 모순된 기록이 된다.\n\n"
        "events 배열의 순서 = 검증 순서다: 각 사건은 자신보다 앞에 나온 사건들만 이미 벌어진 "
        "일로 보고 검증되고, 뒤에 나온 사건은 아직 모른다. 따라서 clear로 기존 상태/관계를 "
        "끝내야만 말이 되는 사건이 있다면, 그 clear를 반드시 그 사건보다 앞에 배치하라 — "
        "예를 들어 '추방이 풀린 뒤 다시 방문했다'는 [추방 해제(clear)] -> [방문(point)] 순서여야 "
        "한다. clear를 맨 뒤에 두면 앞선 사건들이 여전히 열려 있는 기존 상태와 모순되는 것으로 "
        "판단된다.\n\n"
        "duration_effect.predicate: 대상이 없는 개인 상태라면 아래 등록된 id 중 하나를 써라:\n"
        f"{status_effect_options}\n"
        "대상이 있는 관계라면, 이미 등록된 관계형 predicate 목록을 먼저 확인하고 상황에 맞는 게 "
        f"있으면 재사용하라:\n{relational_predicate_options}\n"
        "마땅히 재사용할 것이 없을 때만 새로운 predicate 이름을 자유롭게 만들어라 — 새 이름은 "
        "이후 별도 확인 절차를 거치므로 지어내는 것 자체는 괜찮다. clear일 때는 새 이름을 짓지 "
        "말고 반드시 닫으려는 기존 기록의 predicate를 그대로 재사용하라.\n\n"
        "각 point 사건에는 notes(실제 있었던 일을 서술하는 완결된 한국어 문장 — 이 문장은 이후 "
        "세계관 규칙/설정 모순 검증에 그대로 쓰이므로, 검증 가능하도록 구체적으로 서술하라)와 "
        "involved_entities(관련된 entity_id 목록)를 채워라. 각 duration 사건에는 notes와 "
        "duration_effect(entity, predicate, target, action, start_year, end_year — action에 "
        "따라 위 설명대로 채움)를 채워라.\n\n"
        "=== 기존 장소/사물/세력 활용 ===\n"
        "point 사건이 특정 장소에서 벌어진다면, 아래 목록에 있는 경우에만 location에 해당 "
        "entity_id를 채워라(장소가 아니면 location은 항상 null). 아래 목록의 사물/세력도 "
        "서사에 자연스럽게 등장시켜도 좋다 — 등장시켰다면 involved_entities에 반드시 포함시켜라 "
        "(포함시키지 않으면 그 엔티티 쪽에서는 이 사건이 전혀 기록되지 않는다). 목록에 없는 "
        "장소/사물/세력은 지어내지 마라 — 서사에 특별히 필요하지 않다면 억지로 아무거나 "
        "골라 넣지 마라. 특히, 위 '엔티티 정보'에 태그된 인물이 이미 어떤 사물을 소유하고 "
        "있거나(예: 소유 기간이 이 서사의 연도에 걸쳐 유효한 'owns' 기록) 어떤 세력에 소속되어 "
        "있다는 기록이 있고, 요청 내용이 '검', '명검', '단체' 같은 뭉뚱그린 표현으로 그런 대상을 "
        "가리킬 수 있는 상황이라면 — 지어낸 일반 표현 대신 그 구체적인 기존 entity_id를 지목해서 "
        "써라(예: 그냥 '명검'이 아니라 실제 소유 중인 '엑스칼리버'로).\n"
        f"등록된 장소/사물/세력 목록:\n{backdrop_block}\n"
        f"{new_entity_block}"
        f"{single_year_instruction}{feedback_block}{supplement_block}\n\n"
        "아래 JSON 형식으로만 답하라 (다른 설명 금지):\n"
        "{\n"
        '  "natural_event_count": 정수 (이 서사에 이상적인 사건 개수, 제약 없이 판단),\n'
        '  "new_entities": [\n'
        "    {\n"
        '      "tag": "이 응답 안에서만 쓰는 임시 식별자",\n'
        '      "category": "허용된 카테고리 중 하나",\n'
        '      "fields": {"name": "...", "필수 필드": "...", "notes": "선택, 짧은 설명"}\n'
        "    }\n"
        "  ] (새 엔티티 생성이 허용되지 않았다면 항상 빈 배열),\n"
        '  "events": [\n'
        "    {\n"
        '      "event_type": "point 또는 duration",\n'
        '      "notes": "한국어 문장",\n'
        '      "involved_entities": ["entity_id 또는 new_entities의 tag", ...],\n'
        '      "year": "point일 때만, 정수 또는 null",\n'
        '      "location": "point일 때만, 등록된 장소의 entity_id 또는 null",\n'
        '      "duration_effect": {\n'
        '        "entity": "entity_id 또는 tag", "predicate": "...", '
        '"target": "entity_id, tag, 또는 null", '
        '"action": "set, clear, set_closed 중 하나",\n'
        '        "start_year": "set/set_closed일 때 필수, clear일 때는 null 가능, 정수 또는 null",\n'
        '        "end_year": "clear/set_closed일 때 필수, set일 때는 null, 정수 또는 null"\n'
        "      } 또는 null (duration일 때만)\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    # Tagged entities + every backdrop entity Creator was actually shown as
    # freely referenceable (backdrop_ids, above) — NOT tagged entities
    # alone. A location/artifact/faction is legitimately usable without
    # ever being tagged in the *current* request (e.g. reusing
    # loc_은빛도시_지하감옥 from an already-existing imprisonment record
    # tagged only in some earlier, separate request); validating against
    # just resolved_entities rejected exactly this as if it were a
    # hallucinated id, when it was a perfectly real, intentionally-always-
    # usable one (caught via direct repro, not anticipated when this
    # validation was first added).
    tagged_and_backdrop_ids = set(resolved_entities.values()) | set(backdrop_ids)

    # Retried as a whole (invoke -> resolve -> build -> validate), not just
    # the raw invoke+parse _invoke_llm_json already retries on its own —
    # this catches a *structurally valid* JSON response that still
    # references an entity_id nothing actually resolves to: an LLM
    # occasionally hallucinates a near-miss id for a rare custom name (the
    # "쟩" vs the real "데이비드" case _matching_tagged_entity guards against for
    # new_entities specifically) and can just as easily drop that same
    # typo'd id straight into a duration_effect/involved_entities reference
    # without ever declaring it as a new entity at all — nothing downstream
    # would ever notice a bogus string flowing through as if it were real.
    # _autocorrect (below) fixes an obvious one-character slip like that in
    # place rather than failing the whole attempt over it; only a
    # reference with no close match at all still falls through to
    # bad_reference and a full recompose.
    last_error = None
    for _ in range(2):
        data = _invoke_llm_json(prompt)
        new_entities, tag_to_id = _resolve_new_entities(data, allowed_new_categories, resolved_entities)
        _autopromote_undeclared_ids(data, allowed_new_categories, tag_to_id, new_entities)
        all_valid_ids = tagged_and_backdrop_ids | set(tag_to_id.values())

        def _autocorrect(ref):
            # Fixes an obvious one-character hallucinated slip (e.g.
            # "char_쟩" for the real, already-valid "char_데이비드") in place
            # instead of failing the whole attempt over it — only a
            # reference with no close match at all falls through to
            # bad_reference below.
            if ref is None or ref in all_valid_ids:
                return ref
            return _closest_valid_id(ref, all_valid_ids) or ref

        events = []
        bad_reference = None
        for e in (data.get("events") or [])[:MAX_EVENTS]:
            involved = [_autocorrect(_remap_tags(v, tag_to_id)) for v in (e.get("involved_entities") or [])]
            location = _autocorrect(_remap_tags(e.get("location"), tag_to_id))
            duration_effect = e.get("duration_effect")
            if duration_effect:
                duration_effect = dict(duration_effect)
                duration_effect["entity"] = _autocorrect(_remap_tags(duration_effect.get("entity"), tag_to_id))
                duration_effect["target"] = _autocorrect(_remap_tags(duration_effect.get("target"), tag_to_id))
                if duration_effect.get("action") in ("clear", "set_closed") and not duration_effect.get("target"):
                    # The narrative closing a status (e.g. "석방되었다") rarely
                    # re-mentions the target it was recorded against (e.g. the
                    # prison), so the LLM naturally leaves target null even
                    # though hard_check.check_duration_closure_conflict and
                    # inspect_draft's closed_predicates both key on the full
                    # (entity, predicate, target) triple and silently no-op
                    # instead of actually closing anything on a target
                    # mismatch (None != loc_은빛도시_지하감옥) — observed via
                    # direct user report: a release event "passed" but the very
                    # next event was still rejected as if still imprisoned.
                    # Fill it in from whatever's actually open in storage for
                    # this (entity, predicate), same "resolve before it can go
                    # wrong" pattern as _backdrop_entity_ids/temporal filters.
                    duration_effect["target"] = _resolve_open_target(
                        duration_effect.get("entity"), duration_effect.get("predicate")
                    )
            # Folded into involved_entities here, not left as a bare field —
            # this is what actually gets the location a reciprocal event_ids
            # pointer (via archivist's own pointer registration), the same way
            # a location tagged directly in the normal chat pipeline gets one.
            # Without this, the location field alone would still render as a
            # clickable link on the event's own page (Phase 10 patch 20 renders
            # any set reference field), but the location's *own* page would
            # never show this event back — a one-directional link, not the
            # real thing. New entities referenced only via duration_effect (not
            # involved_entities) get the same treatment for the same reason.
            if location and location not in involved:
                involved.append(location)
            for extra in (
                (duration_effect or {}).get("entity"),
                (duration_effect or {}).get("target"),
            ):
                if extra and extra not in involved:
                    involved.append(extra)

            for ref in involved:
                if ref not in all_valid_ids:
                    bad_reference = ref
                    break
            if bad_reference:
                break

            events.append(
                DraftEvent(
                    event_type=e.get("event_type", "point"),
                    notes=e.get("notes", ""),
                    involved_entities=involved,
                    year=e.get("year"),
                    location=location,
                    duration_effect=duration_effect,
                )
            )

        if bad_reference is None:
            return NarrativeDraft(
                events=events,
                new_entities=new_entities,
                natural_event_count=data.get("natural_event_count") or len(events),
            )
        last_error = ValueError(f"존재하지 않는 entity_id를 참조했습니다: {bad_reference!r}")

    raise last_error


# ---------------------------------------------------------------------------
# Inspector — reuses Step 4 (check_rule_and_notes) + Step 5 (hard_check)
# unmodified, walking the draft sequentially so a later event's check can
# see earlier events in the same draft as context, even though nothing is
# written to storage until the whole batch is approved and saved.
# ---------------------------------------------------------------------------

@dataclass
class InspectionResult:
    approved: bool
    reason: str | None = None  # combined human-readable feedback, for Creator's retry or the final rejection message
    failed_event_index: int | None = None


def _event_involved(event: DraftEvent, fallback: list) -> list:
    if event.event_type == "duration" and event.duration_effect:
        involved = [
            v for v in (event.duration_effect.get("entity"), event.duration_effect.get("target")) if v
        ]
        if involved:
            return involved
    return event.involved_entities or fallback


def inspect_draft(resolved_entities: dict, draft: NarrativeDraft) -> InspectionResult:
    """Stops at the first rejected event — Creator retries the whole batch
    (spec: whole-batch retry, not per-event patching), so nothing is gained
    by continuing to check events past the first failure.

    An empty draft.events is a rejection, not a trivial pass (the for loop
    below would otherwise never execute and fall through to approved=True)
    — observed in practice: composing against a year constraint that
    conflicts with an established relational fact (e.g. asking for a 2030s
    scene between two characters already on record as not meeting until
    2079) can make the LLM give up and return zero events for an attempt
    instead of erroring, which used to silently "succeed" with nothing to
    save and no failure reason ever shown."""
    if not draft.events:
        return InspectionResult(approved=False, reason="이번 시도에서 생성된 사건이 없습니다.")

    entity_ids = list(resolved_entities.values())
    hard_rule_docs = rag_check._get_hard_rule_texts()
    approved_context_lines = []  # this draft's own already-approved events' notes
    approved_years = {}  # entity_id -> [year, ...] already used earlier in this draft
    # (entity_id, predicate) pairs closed by an earlier event in this same
    # draft (a `clear`/`set_closed` duration_effect) — nothing is saved yet,
    # so storage still shows the original record open; without tracking
    # this separately, a later event's context kept getting the stale
    # record's confirmatory [활성] tag even after the release event had
    # already been approved earlier in the same draft (observed: a prison
    # release event passed, but the very next event was still rejected for
    # supposedly still being imprisoned).
    closed_predicates = set()

    for i, event in enumerate(draft.events):
        involved = _event_involved(event, entity_ids)
        # A "clear" action may carry only end_year (start_year belongs to
        # the *existing* record being closed, not this one) — fall back to
        # it so the check still has a year to annotate/gate against.
        event_year = event.year if event.event_type == "point" else (event.start_year or event.end_year)
        candidate_years = [y for y in (event.year, event.start_year, event.end_year) if y is not None]

        for entity_id in involved:
            category = schema.category_from_id(entity_id)
            if category is None:
                continue
            extra_years = candidate_years + approved_years.get(entity_id, [])
            conflicts = hard_check.run_hard_checks(category, entity_id, extra_years=extra_years)
            blocking = [c for c in conflicts if c.severity == "blocking"]
            if blocking:
                reason = (
                    f"{i + 1}번째 사건(\"{event.notes}\")이 하드체크에 위반됩니다: "
                    + "; ".join(c.reason for c in blocking)
                )
                return InspectionResult(approved=False, reason=reason, failed_event_index=i)

        if event.event_type == "duration":
            closure_conflict = hard_check.check_duration_closure_conflict(event.duration_effect)
            if closure_conflict is not None:
                reason = (
                    f"{i + 1}번째 사건(\"{event.notes}\")이 하드체크에 위반됩니다: "
                    f"{closure_conflict.reason}"
                )
                return InspectionResult(approved=False, reason=reason, failed_event_index=i)

        judgments = rag_check.check_rule_and_notes(
            involved, event.notes, hard_rule_docs, event_year,
            extra_context=approved_context_lines, closed_predicates=closed_predicates,
        )
        if judgments:
            reasons = "; ".join(f"[{j.type}] {j.reason}" for j in judgments)
            reason = f"{i + 1}번째 사건(\"{event.notes}\")이 검증에 실패했습니다: {reasons}"
            return InspectionResult(approved=False, reason=reason, failed_event_index=i)
        print(f"[inspector] {i + 1}번째 사건 검증 통과")

        # One combined line for every involved entity, not one line per
        # entity (token-diet pass) — a 2+ entity event used to repeat its
        # own notes text once per participant, the same duplication
        # _entity_context_block had, just accumulating across a whole
        # draft's worth of events instead of a whole entity's history.
        if involved:
            entity_label = ", ".join(involved)
            approved_context_lines.append(
                f"{entity_label}의 관련 기록(이번 초안 {i + 1}번째 사건): {event.notes}"
            )
        for entity_id in involved:
            if candidate_years:
                approved_years.setdefault(entity_id, []).extend(candidate_years)

        if event.event_type == "duration" and event.duration_effect:
            action = event.duration_effect.get("action", "set")
            if action in ("clear", "set_closed"):
                closer_entity = event.duration_effect.get("entity")
                predicate = event.duration_effect.get("predicate")
                if closer_entity and predicate:
                    # Keyed by (entity, predicate, target), not just
                    # (entity, predicate) — the same predicate can be open
                    # toward multiple different targets at once (X friends
                    # with both A and B), and closing X-A must not also
                    # suppress X-B's own still-genuinely-active [활성] tag
                    # (caught via direct repro, same family of bug as
                    # check_duration_closure_conflict conflating separate
                    # episodes of the same predicate).
                    closed_predicates.add((closer_entity, predicate, event.duration_effect.get("target")))

    return InspectionResult(approved=True)


# ---------------------------------------------------------------------------
# Reflection loop — Creator drafts, Inspector checks, repeat on rejection
# ---------------------------------------------------------------------------

MAX_RETRIES = 3  # spec: "3~5회" — lowered from 4 (token-diet pass): each
# retry re-drafts and re-checks the whole batch from scratch (see
# inspect_draft's own docstring), so worst-case cost scales directly with
# this number.


@dataclass
class ReflectionResult:
    draft: NarrativeDraft
    approved: bool
    attempts: int
    last_reason: str | None = None  # set only when approved=False — the final attempt's rejection reason


def _print_draft_log(attempt: int, draft: NarrativeDraft) -> None:
    """A clean, demo-legible summary of one attempt — just the generated
    story and (via the caller, right after inspect_draft) its rejection
    reason if any. rag_check's own per-check context/verdict prints are the
    right level of detail for debugging the checks themselves, but they're
    noise for watching the reflection loop from the outside: same raw
    entity context dumped again on every single event of every attempt,
    with nothing marking where one attempt ends and the retry begins."""
    print(f"[creator] {attempt}번째 시도")
    for event in draft.events:
        if event.event_type == "point":
            print(f"  [point, {event.year}년] {event.notes}")
        else:
            print(f"  [duration] {event.notes}")


def run_reflection_loop(
    resolved_entities: dict,
    request_text: str,
    lower: int,
    upper: int,
    supplement: str | None = None,
    first_draft: NarrativeDraft | None = None,
    allowed_new_categories: set | None = None,
) -> ReflectionResult:
    """Never silently gives up (spec section E): on exhausting MAX_RETRIES,
    returns the *last* attempted draft alongside why it was rejected, so the
    caller can show both to the user for a manual decision rather than just
    reporting failure. `supplement` (an optional [Redo] instruction) stays
    constant across every retry within this one call; `feedback` (Inspector's
    rejection reason) changes attempt to attempt, feeding forward so Creator
    doesn't blindly repeat the same mistake.

    `first_draft`, when given, is used as attempt 1 instead of composing a
    fresh one — lets a caller that already had to call compose_narrative
    once for its own reasons (creator_session's single-year count-mismatch
    check draws its own first draft to inspect natural_event_count before
    the user has even confirmed a final year window) feed it in here rather
    than paying for a redundant duplicate composition."""
    feedback = None
    draft = first_draft
    start_attempt = 1
    if draft is not None:
        _print_draft_log(1, draft)
        result = inspect_draft(resolved_entities, draft)
        if result.approved:
            print(f"[inspector] 1번째 시도 전체 승인 — 저장 대기")
            return ReflectionResult(draft=draft, approved=True, attempts=1)
        print(f"[creator] 반려 사유: {result.reason}")
        feedback = result.reason
        start_attempt = 2

    for attempt in range(start_attempt, MAX_RETRIES + 1):
        draft = compose_narrative(
            resolved_entities,
            request_text,
            lower,
            upper,
            feedback=feedback,
            supplement=supplement,
            allowed_new_categories=allowed_new_categories,
        )
        _print_draft_log(attempt, draft)
        result = inspect_draft(resolved_entities, draft)
        if result.approved:
            print(f"[inspector] {attempt}번째 시도 전체 승인 — 저장 대기")
            return ReflectionResult(draft=draft, approved=True, attempts=attempt)
        print(f"[creator] 반려 사유: {result.reason}")
        feedback = result.reason

    return ReflectionResult(draft=draft, approved=False, attempts=MAX_RETRIES, last_reason=feedback)
