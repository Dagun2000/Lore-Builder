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
from dataclasses import dataclass, field

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
    _generator: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Generator equivalents of mapping.py's interactive helpers
# ---------------------------------------------------------------------------

def _create_new_entity_gen(
    session: PipelineSession, inferred_category: str, tag: str, context_sentence: str, year: int
):
    """Phase 9 patch A: category is confirmed (or corrected) up front,
    together with the name, before anything else — misclassifying the
    *category* (e.g. a person mistaken for an item) was the real risk, not
    the name, which the LLM rarely gets wrong. "저장 후 계속" commits
    category+name only, leaving every other field blank; "편집" continues
    into the full field form (required forced, optional included); "취소"
    aborts the whole input via EntityCreationCancelled."""
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

    entity_id = mapping.generate_entity_id(category, fields.get("name", tag))

    if action == "save":
        storage.save_entity(category, entity_id, fields)
        storage.save_to_chroma(entity_id, context_sentence, {"category": category, "tag": tag})
        return entity_id

    # action == "edit": full field form below (required forced, optional
    # included) — this is the only path that still asks about terminal
    # status, since that too is a form of "editing" the new character.
    if category == "character" and mapping.infer_terminal_status(context_sentence):
        answer = yield PendingDecision(
            "entity_terminal_status",
            {"tag": tag, "entity_id": entity_id, "year": year},
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
    return entity_id


def _resolve_entity_gen(session: PipelineSession, tag: str, context_sentence: str, year: int):
    category = mapping.infer_category(tag, context_sentence)
    exact_matches, partial_matches = mapping.find_existing_matches(tag, category)

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        choice = yield PendingDecision(
            "entity_candidates",
            {"tag": tag, "candidates": exact_matches, "allow_create": False},
            {"tag": tag},
        )
        return choice

    if partial_matches:
        choice = yield PendingDecision(
            "entity_candidates",
            {"tag": tag, "candidates": partial_matches, "allow_create": True},
            {"tag": tag},
        )
        if choice == CREATE_NEW:
            entity_id = yield from _create_new_entity_gen(session, category, tag, context_sentence, year)
            return entity_id
        return choice

    entity_id = yield from _create_new_entity_gen(session, category, tag, context_sentence, year)
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
        return False  # "수정" / "취소" / anything else

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
        return False  # "수정" / "취소" / anything else

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
    primary_year = parsed.years[0]

    resolved_entities = {}
    for tag in parsed.tags:
        try:
            entity_id = yield from _resolve_entity_gen(session, tag, parsed.raw_text, primary_year)
        except EntityCreationCancelled as exc:
            return {
                "status": "cancelled",
                "stage": "entity_resolution",
                "tag": exc.tag,
                "message": f"'{exc.tag}' 엔티티 생성이 취소되었습니다. 입력을 다시 작성해주세요.",
            }
        resolved_entities[tag] = entity_id
        session.resolved_entities = dict(resolved_entities)

    inferred_event = inference.infer_event(resolved_entities, parsed.raw_text, parsed.years)
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
        extra_years = list(parsed.years) if is_present else None
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

    diff = archivist.build_diff(parsed, resolved_entities, inferred_event)
    if isinstance(diff, archivist.ConfirmationNeeded):
        # No partial diff exists to fall back to — ConfirmationNeeded means
        # no single coherent record could be built at all. "계속 진행" can
        # only mean "acknowledge and stop here without saving anything" (the
        # user re-submits as separate, clearer inputs); "취소" reaches the
        # same outcome, just labeled as an explicit cancel rather than a
        # shrug.
        proceed = yield PendingDecision("multi_event_warning", {"reason": diff.reason}, {})
        if proceed:
            return {
                "status": "no_changes",
                "stage": "confirmation",
                "resolved_entities": resolved_entities,
                "message": diff.reason,
            }
        return {
            "status": "cancelled",
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
