"""Pipeline state machine — Phase 8.

main.run_pipeline (Phase 5) and mapping.resolve_entity/approval.py (Phase 2/5)
are built on blocking input() — fine for the CLI, impossible for a GUI that
must redraw between one function call and the next. This module re-expresses
the same pipeline as something that can pause at a decision point and resume
later from a plain (session_id, response) pair.

Design: the pipeline body is written as a single Python generator that
`yield`s a PendingDecision at every point the old code called `_prompt`, and
receives the human's answer back via `.send()`. Everything that ISN'T
interactive (LLM calls, hard/RAG checks, diff assembly, storage writes) is
reused as-is from schema/storage/hard_check/rag_check/archivist/inference/
mapping — only the handful of functions in mapping.py that used to call
`_prompt`/`print` directly (candidate selection, name entry, terminal-status
confirmation, required-field collection) and approval.py's three review
loops have generator equivalents here. `main.py` becomes a thin adapter that
renders each PendingDecision as the same CLI prompt it always was; a future
GUI would render the same payload as widgets instead.

Sessions live in an in-memory dict — local single-user tool, no persistence
needed, losing them on restart is fine.
"""

import uuid
from dataclasses import dataclass, field, replace

from . import archivist, hard_check, inference, mapping, parser, rag_check, schema, storage

CREATE_NEW = "신규 생성"  # sentinel entity_candidates response: build a new entity instead

_STAGE_BY_DECISION = {
    "entity_candidates": "resolving_entities",
    "entity_category_and_name": "resolving_entities",
    "entity_terminal_status": "resolving_entities",
    "entity_required_field": "resolving_entities",
    "hard_check_warning": "hard_checking",
    "rag_judgment": "rag_checking",
    "multi_event_warning": "confirming_event",
    "diff_review": "reviewing_diff",
}

# Fields never offered on the new-entity confirm/edit screens: name has its
# own dedicated step, and event_ids is system-managed (archivist writes
# pointers there directly — a raw comma-separated form value would corrupt
# that shape).
_NOT_EDITABLE_ON_CREATE = {"name", "event_ids"}


class EntityCreationCancelled(Exception):
    """Raised when the user picks "취소" on the new-entity confirmation
    screen (Phase 9 patch A) — unwinds all the way out of _pipeline_generator
    via the yield-from chain, since cancelling entity creation means this
    whole input can't be processed, not just this one tag."""

    def __init__(self, tag: str):
        super().__init__(tag)
        self.tag = tag


def _field_defs_for_widget(category: str) -> list:
    """Full field-def payload (name/type/required/options/ref_category) for
    every field in `category` that the new-entity edit screen and the
    entity-detail edit screen can both render with the same widget-mapping
    logic (Phase 9 patch B) — reference -> selectbox of existing entities,
    enum -> selectbox of schema options, everything else by primitive type."""
    return [
        {
            "name": f["name"],
            "type": f["type"],
            "required": bool(f.get("required")),
            "options": f.get("options"),
            "ref_category": f.get("ref_category"),
        }
        for f in schema.get_fields(category)
        if f["name"] not in _NOT_EDITABLE_ON_CREATE
    ]

_SESSIONS: dict = {}


@dataclass
class PendingDecision:
    decision_type: str
    payload: dict
    context: dict = field(default_factory=dict)


@dataclass
class PipelineSession:
    session_id: str
    user_input: str
    stage: str = "parsing"
    resolved_entities: dict = field(default_factory=dict)
    inferred_event: object = None
    rag_judgments: list = field(default_factory=list)
    hard_check_conflicts: list = field(default_factory=list)
    diff: list = field(default_factory=list)
    diff_approved: list = field(default_factory=list)
    pending_decision: "PendingDecision | None" = None
    result: dict | None = None
    # Phase 10 patch 4 (J): which entity_ids _create_new_entity_gen actually
    # storage.save_entity'd during this run — lets _pipeline_generator tell
    # "a brand-new entity's attributes were saved" apart from "every tag
    # resolved to something that already existed, nothing was written",
    # which otherwise both look identical (an empty remaining_years pool)
    # from the outside.
    newly_created_entities: list = field(default_factory=list)
    _generator: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Generator equivalents of mapping.py's interactive helpers
# ---------------------------------------------------------------------------

def _create_new_entity_gen(
    session: PipelineSession,
    inferred_category: str,
    tag: str,
    context_sentence: str,
    year: int,
    remaining_years: list,
):
    """Phase 9 patch A: category is confirmed (or corrected) up front,
    together with the name, before anything else — misclassifying the
    *category* (e.g. a person mistaken for an item) was the real risk, not
    the name, which the LLM rarely gets wrong. "저장 후 계속" commits
    category+name plus any auto-filled attributes (see below), skipping only
    the *optional* fields — required fields are never skippable (Phase 10
    patch 2, C); "편집" continues into the full field form (required forced,
    optional included); "취소" aborts the whole input via
    EntityCreationCancelled."""
    category = inferred_category

    while True:
        field_names = {f["name"] for f in schema.get_fields(category)}
        has_name_field = "name" in field_names

        response = yield PendingDecision(
            "entity_category_and_name",
            {
                "tag": tag,
                "inferred_category": inferred_category,
                "categories": schema.list_categories(),
                "has_name_field": has_name_field,
                "default_name": tag,
            },
            {"tag": tag},
        )
        response = response or {}
        category = response.get("category") or category
        action = response.get("action")

        if action == "cancel":
            raise EntityCreationCancelled(tag)
        if action in ("save", "edit"):
            break
        # Anything unrecognized (e.g. a stray empty response): re-show the
        # same confirmation rather than silently guessing an action.

    field_names = {f["name"] for f in schema.get_fields(category)}
    fields = {}
    if "name" in field_names:
        fields["name"] = response.get("name") or tag

    # Phase 10 patch 4 (I): re-check for an existing entity right before
    # minting a new id — covers both "user corrected the category" and "user
    # retyped a different name here" (_resolve_entity_gen's search already
    # covered the original tag/inferred-category pair, but not whatever the
    # user might change on this screen). Without this, a misclassified "밥"
    # that the user manually fixes to "character" still produced a duplicate
    # char_밥_2 instead of resolving to the existing char_밥.
    search_name = fields.get("name", tag)
    exact, _partial = mapping.find_existing_matches(search_name, category)
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        choice = yield PendingDecision(
            "entity_candidates", {"tag": tag, "candidates": exact, "allow_create": False}, {"tag": tag}
        )
        return choice

    entity_id = mapping.generate_entity_id(category, fields.get("name", tag))

    # Phase 10 patch 2 (A) / patch 3 (E): split the sentence's content about
    # this brand-new entity into lifecycle fields, a leftover time-bound
    # event (untouched here — Step 3 still handles that), and a persistent
    # trait/rule statement that belongs in `notes` (never an event, never
    # requires a year). An already-existing entity never reaches this
    # generator (see _resolve_entity_gen), so its fields/notes can never be
    # silently changed from chat. A *bare* consumed year (nothing more than
    # the fact itself) is popped from `remaining_years` so Step 3's event
    # judgment never sees it (otherwise a founding-year mention sitting next
    # to an unrelated event year falsely looks like two separate events) —
    # but a *narrative* consumed year (patch 4, K: the fact came with actual
    # circumstance/interaction, e.g. "110년에 술을 먹고 싸우다가 죽었다")
    # stays in the pool, because filling death_year here doesn't replace the
    # point event Step 3 still needs to build for that same year.
    attrs = inference.infer_new_entity_attributes(category, tag, context_sentence, list(remaining_years))
    for name, value in attrs["attributes"].items():
        if fields.get(name) is None:
            fields[name] = value
    narrative_years = set(attrs.get("narrative_years") or [])
    for consumed_year in attrs["consumed_years"]:
        if consumed_year in narrative_years:
            continue
        if consumed_year in remaining_years:
            remaining_years.remove(consumed_year)
    if attrs.get("notes") and fields.get("notes") is None:
        fields["notes"] = attrs["notes"]

    if action == "save":
        # "저장 후 계속" skips only the *optional* fields — required fields
        # (schema `required: true`) can never be skipped, no matter how many
        # a category has (Phase 10 patch 2, C: this used to only force the
        # first one via the name field, silently dropping the rest).
        editable_fields = [
            f for f in _field_defs_for_widget(category)
            if f["required"] and fields.get(f["name"]) is None
        ]
    else:
        # action == "edit": full field form below (required forced, optional
        # included) — also the only path that still asks about terminal
        # status, as a fallback for softer language the attribute extractor
        # above wasn't confident enough to fill in directly.
        if (
            category == "character"
            and fields.get("death_year") is None
            and year is not None
            and mapping.infer_terminal_status(context_sentence)
        ):
            answer = yield PendingDecision(
                "entity_terminal_status",
                {"tag": tag, "entity_id": entity_id, "year": year, "field_name": "death_year"},
                {"tag": tag},
            )
            if answer == "예":
                fields["death_year"] = year
            elif isinstance(answer, dict):
                fields.update(answer.get("수정") or {})
            # "아니오" (or anything else unrecognized): no death_year set.

        editable_fields = [
            f for f in _field_defs_for_widget(category) if fields.get(f["name"]) is None
        ]

    while editable_fields:
        response = yield PendingDecision(
            "entity_required_field",
            {"category": category, "fields": editable_fields},
            {"tag": tag},
        )
        response = response or {}
        still_missing = []
        for f in editable_fields:
            raw_value = response.get(f["name"])
            if raw_value is None or raw_value == "" or raw_value == []:
                if f["required"]:
                    still_missing.append(f)
                continue
            if isinstance(raw_value, str):
                # CLI submits raw text for every field; coerce like any
                # other typed-from-a-string entry point (schema.coerce_value).
                field_def = next(fd for fd in schema.get_fields(category) if fd["name"] == f["name"])
                fields[f["name"]] = schema.coerce_value(field_def, raw_value)
            else:
                # A GUI widget (checkbox/number_input/selectbox/...) already
                # returns the correctly-typed Python value — use it as-is.
                fields[f["name"]] = raw_value
        editable_fields = still_missing

    storage.save_entity(category, entity_id, fields)
    storage.save_to_chroma(entity_id, context_sentence, {"category": category, "tag": tag})
    session.newly_created_entities.append(entity_id)
    return entity_id


def _resolve_entity_gen(session: PipelineSession, tag: str, context_sentence: str, remaining_years: list):
    """`remaining_years` is the whole input's year pool, shared and mutated
    across every tag in this input — a new entity created here may consume
    some of it as lifecycle attributes (see _create_new_entity_gen), which
    is why this is threaded by reference instead of a single primary year.

    Phase 10 patch 4 (I): name is searched across *every* name-bearing
    category before category is ever inferred — an existing entity must
    always be findable by name regardless of whether the (LLM-guessed)
    category is right, since gating the search behind that guess is what
    let a misclassified "밥" (an existing character, misread as a race)
    produce a duplicate instead of resolving to char_밥. Category inference
    only runs once this search comes up completely empty."""
    exact_matches, partial_matches = mapping.find_existing_matches_any_category(tag)
    year = remaining_years[0] if remaining_years else None

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        choice = yield PendingDecision(
            "entity_candidates",
            {"tag": tag, "candidates": exact_matches, "allow_create": False},
            {"tag": tag},
        )
        return choice

    category = mapping.infer_category(tag, context_sentence)

    if partial_matches:
        choice = yield PendingDecision(
            "entity_candidates",
            {"tag": tag, "candidates": partial_matches, "allow_create": True},
            {"tag": tag},
        )
        if choice == CREATE_NEW:
            entity_id = yield from _create_new_entity_gen(
                session, category, tag, context_sentence, year, remaining_years
            )
            return entity_id
        return choice

    entity_id = yield from _create_new_entity_gen(
        session, category, tag, context_sentence, year, remaining_years
    )
    return entity_id


# ---------------------------------------------------------------------------
# Generator equivalents of approval.py's three review loops
# ---------------------------------------------------------------------------

def _review_hard_check_conflicts_gen(conflicts: list):
    blocking = [c for c in conflicts if c.severity == "blocking"]
    if blocking:
        return False  # immediate rejection, never a decision point

    for c in [c for c in conflicts if c.severity == "warning"]:
        answer = yield PendingDecision(
            "hard_check_warning",
            {"check_type": c.check_type, "entity_id": c.entity_id, "reason": c.reason},
            {},
        )
        if answer == "그래도 저장":
            if c.check_type == "lifespan":
                storage.save_entity("character", c.entity_id, {"lifespan_check_ack": True})
            continue
        return False  # "취소" (or anything else unrecognized)

    return True


def _review_rag_judgments_gen(judgments: list):
    for j in judgments:
        if j.type == "clears_status":
            continue  # informational auto-accept, never a decision point

        answer = yield PendingDecision(
            "rag_judgment",
            {
                "judgment_type": j.type,
                "reason": j.reason,
                "entity_id": j.entity_id,
                "status_effect_id": j.status_effect_id,
            },
            {},
        )
        if answer == "그래도 저장":
            continue
        return False  # "취소" (or anything else unrecognized)

    return True


def _review_diff_gen(diff: list):
    """One bundled decision for the whole diff, not one per ChangeItem.
    Phase 10's diffs are always "one primary timeline record + pointer/cache
    updates that belong to it" (never several unrelated changes at once, the
    way old relationship-heavy diffs sometimes were) — so there's nothing to
    approve item-by-item. Show the primary record plus who else gets
    touched, and it's just 저장 (apply everything) or 취소 (apply nothing);
    no per-field edit here — if something in `affected_entities` is wrong,
    that's an entity-resolution problem to fix by re-submitting the input,
    not something to patch mid-review."""
    if not diff:
        return []

    primary = next((c for c in diff if c.category == "timeline"), diff[0])
    affected = [c.entity_id for c in diff if c is not primary]

    answer = yield PendingDecision(
        "diff_review",
        {
            "action": primary.action,
            "category": primary.category,
            "entity_id": primary.entity_id,
            "fields": primary.fields,
            "reason": primary.reason,
            "affected_entities": affected,
        },
        {},
    )
    return diff if answer else []


def _apply_diff(approved: list) -> list:
    creates = [c for c in approved if c.action == "create"]
    updates = [c for c in approved if c.action == "update"]

    applied = []
    for item in creates + updates:
        storage.save_entity(item.category, item.entity_id, item.fields)
        if item.body:
            storage.save_to_chroma(item.entity_id, item.body, {"category": item.category})
        applied.append(item)
    return applied


# ---------------------------------------------------------------------------
# Top-level pipeline generator
# ---------------------------------------------------------------------------

def _pipeline_generator(session: PipelineSession):
    parsed = parser.parse_input(session.user_input)
    # Shared, mutated pool: a brand-new entity may consume one of these years
    # as a lifecycle attribute (founded_year, death_year, ...) instead of it
    # becoming part of an event (Phase 10 patch 2, A) — whatever's left after
    # entity resolution is what Step 3+ actually judges as "the event".
    remaining_years = list(parsed.years)

    resolved_entities = {}
    for tag in parsed.tags:
        try:
            entity_id = yield from _resolve_entity_gen(session, tag, parsed.raw_text, remaining_years)
        except EntityCreationCancelled as exc:
            return {
                "status": "cancelled",
                "stage": "entity_resolution",
                "tag": exc.tag,
                "message": f"'{exc.tag}' 엔티티 생성이 취소되었습니다. 입력을 다시 작성해주세요.",
            }
        resolved_entities[tag] = entity_id
        session.resolved_entities = dict(resolved_entities)

    if not remaining_years:
        # Every extracted year was consumed as a (bare) lifecycle attribute
        # during entity creation (e.g. "1900년에 창단된 [세력]은 ..."), or
        # there was no year in the input to begin with — either way there's
        # no year left to anchor a timeline record to, so there's no event
        # to judge/check/diff.
        #
        # Phase 10 patch 4 (J): this used to unconditionally report success
        # here, even when every tag resolved to an *existing* entity and
        # nothing was actually written anywhere (e.g. "[아마조네스 용병단]은
        # 아주 유명하다" about an entity that already exists — chat never
        # edits an existing entity's fields, per the standing boundary, so
        # there was genuinely nothing to save, but the old message claimed
        # "속성 정보가 저장되었습니다" regardless). Only report the
        # attributes-saved status when _create_new_entity_gen actually
        # created something this run; otherwise say so plainly and point at
        # the real edit path.
        if session.newly_created_entities:
            # Phase 10 patch 9 (A): a brand-new entity's fields/notes are a
            # new claim about the world exactly like an event is — they must
            # go through Step 4 before being treated as accepted, even though
            # there's no timeline record to hang the check on. (a bare
            # lifecycle-only creation with no notes/attributes worth checking
            # still runs this — check_rule_violation/check_notes_conflict
            # both no-op harmlessly when there's nothing relevant to compare.)
            rag_judgments = rag_check.run_entity_creation_checks(
                list(resolved_entities.values()), parsed.raw_text
            )
            session.rag_judgments = rag_judgments

            ok = yield from _review_rag_judgments_gen(rag_judgments)
            if not ok:
                return {
                    "status": "rejected",
                    "stage": "rag_check",
                    "resolved_entities": resolved_entities,
                    "rag_judgments": rag_judgments,
                }

            return {
                "status": "entity_only",
                "resolved_entities": resolved_entities,
                "message": "속성 정보가 엔티티에 직접 저장되었고, 별도의 사건 기록은 생성되지 않았습니다.",
            }
        return {
            "status": "no_new_info",
            "resolved_entities": resolved_entities,
            "message": (
                "새로 저장할 내용이 없습니다. 채팅은 신규 사건/엔티티 등록 전용이라 기존 엔티티의 "
                "속성은 바꾸지 않습니다 — 기존 엔티티 정보를 수정하려면 디테일 패널을 이용해주세요."
            ),
        }

    event_parsed = replace(parsed, years=remaining_years)
    primary_year = remaining_years[0]

    inferred_event = inference.infer_event(resolved_entities, parsed.raw_text, remaining_years)
    session.inferred_event = inferred_event

    rag_judgments = rag_check.run_rag_checks(
        list(resolved_entities.values()), parsed.raw_text, primary_year
    )
    session.rag_judgments = rag_judgments

    conflicts = []
    for entity_id in resolved_entities.values():
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        # Only entities the event implies were actually present/alive get
        # this event's year(s) injected as a hard-check constraint — see
        # main.run_pipeline's identical comment for the "digging up a grave"
        # vs "playing together" example this guards against.
        is_present = inferred_event.entity_presence.get(entity_id, True)
        extra_years = list(remaining_years) if is_present else None
        conflicts.extend(hard_check.run_hard_checks(category, entity_id, extra_years=extra_years))
    session.hard_check_conflicts = conflicts

    ok = yield from _review_hard_check_conflicts_gen(conflicts)
    if not ok:
        return {
            "status": "rejected",
            "stage": "hard_check",
            "resolved_entities": resolved_entities,
            "conflicts": conflicts,
        }

    ok = yield from _review_rag_judgments_gen(rag_judgments)
    if not ok:
        return {
            "status": "rejected",
            "stage": "rag_check",
            "resolved_entities": resolved_entities,
            "rag_judgments": rag_judgments,
        }

    diff = archivist.build_diff(event_parsed, resolved_entities, inferred_event)
    if isinstance(diff, archivist.ConfirmationNeeded):
        # No partial diff exists to fall back to — ConfirmationNeeded means
        # no single coherent record could be built at all, and nothing gets
        # saved no matter how the user responds. A "계속 진행"/"취소" choice
        # here used to lead to the exact same no-op outcome either way
        # (just a different status label) — genuinely confusing, since
        # picking "계속 진행" looks like it should do something. This is now
        # a single acknowledgment: the user re-submits as separate, clearer
        # inputs.
        yield PendingDecision("multi_event_warning", {"reason": diff.reason}, {})
        return {
            "status": "no_changes",
            "stage": "confirmation",
            "resolved_entities": resolved_entities,
            "message": diff.reason,
        }

    session.diff = diff
    approved = yield from _review_diff_gen(diff)
    session.diff_approved = approved

    if not approved:
        return {
            "status": "no_changes",
            "resolved_entities": resolved_entities,
            "diff": diff,
            "approved": [],
        }

    applied = _apply_diff(approved)

    # Phase 10 patch 6 (B): a narrow, explicit exception to "chat never
    # edits an existing entity's fields" — an existing entity's terminal
    # field (death_year/disbanded_year/destroyed_year) is the one thing
    # hard_check's terminal-violation logic depends on, and almost nobody
    # sets it by hand in the detail panel; the event narrating that entity's
    # end is already being saved above regardless of the answer here, this
    # only asks whether to *additionally* reflect it on the entity itself.
    # A brand-new entity never reaches this (its own creation flow already
    # handles this — see infer_new_entity_attributes / the entity_terminal_
    # status fallback above), so only pre-existing entities are asked.
    terminal_updates = {}
    for tag, entity_id in resolved_entities.items():
        if entity_id in session.newly_created_entities:
            continue
        if not inferred_event.terminal_entities.get(entity_id):
            continue
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        end_fields = schema.get_fields_with_role(category, "lifecycle_end")
        if not end_fields:
            continue
        end_field = end_fields[0]["name"]
        entity = storage.get_entity(category, entity_id)
        if entity is None or entity.get(end_field) is not None:
            continue  # already set — nothing to reconcile

        answer = yield PendingDecision(
            "entity_terminal_status",
            {"tag": tag, "entity_id": entity_id, "year": primary_year, "field_name": end_field},
            {"tag": tag},
        )
        if answer == "예":
            storage.save_entity(category, entity_id, {end_field: primary_year})
            terminal_updates[entity_id] = {end_field: primary_year}
        elif isinstance(answer, dict):
            update = answer.get("수정") or {}
            if update:
                storage.save_entity(category, entity_id, update)
                terminal_updates[entity_id] = update
        # "아니오" (or anything else unrecognized): field left untouched.

    return {
        "status": "saved",
        "parsed": parsed,
        "resolved_entities": resolved_entities,
        "inferred_event": inferred_event,
        "rag_judgments": rag_judgments,
        "conflicts": conflicts,
        "diff": diff,
        "approved": approved,
        "applied": applied,
        "terminal_updates": terminal_updates,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _advance(session: PipelineSession, send_value=None, is_start: bool = False) -> PipelineSession:
    gen = session._generator
    try:
        pending = next(gen) if is_start else gen.send(send_value)
    except StopIteration as stop:
        result = stop.value or {}
        session.result = result
        session.pending_decision = None
        session.stage = "done" if result.get("status") == "saved" else "aborted"
        return session
    except ValueError as exc:
        if not is_start:
            raise
        # Only parser.parse_input can raise before the first yield — mirrors
        # main.run_pipeline's own try/except ValueError around that call.
        session.result = {"status": "error", "stage": "parse", "message": str(exc)}
        session.pending_decision = None
        session.stage = "aborted"
        return session

    session.pending_decision = pending
    session.stage = _STAGE_BY_DECISION.get(pending.decision_type, session.stage)
    return session


def start_session(user_input: str) -> PipelineSession:
    session = PipelineSession(session_id=str(uuid.uuid4()), user_input=user_input)
    session._generator = _pipeline_generator(session)
    _SESSIONS[session.session_id] = session
    return _advance(session, is_start=True)


def resume_session(session_id: str, decision_response) -> PipelineSession:
    session = _SESSIONS[session_id]
    if session.pending_decision is None:
        raise ValueError(f"세션 {session_id}에는 응답을 기다리는 결정이 없습니다.")
    return _advance(session, send_value=decision_response)
