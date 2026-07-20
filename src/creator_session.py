"""Creator interactive state machine (Phase 10 patch 22).

Mirrors pipeline_session.py's generator-based PendingDecision/.send()
pattern — own PendingDecision/CreatorSession dataclasses, own in-memory
_SESSIONS, own start_session/resume_session — rather than sharing
pipeline_session's harness. Deliberately separate (spec section A): normal
chat and Creator branch into completely different entry functions, no
shared session type, no pattern-detection guessing which flow an input
belongs to.

Two decision types are reused verbatim from pipeline_session (same
decision_type string and payload shape) so app.py's existing renderers
handle both flows without new GUI code: "entity_candidates" (tag
disambiguation) and "new_relational_predicate" (patch 16's registry
confirmation).
"""

import uuid
from dataclasses import dataclass, field

from . import archivist, creator, inference, mapping, parser, schema, storage

_SESSIONS: dict = {}

_STAGE_BY_DECISION = {
    "entity_candidates": "resolving_entities",
    "creator_year_confirm": "confirming_year",
    "creator_count_mismatch": "confirming_year",
    "new_relational_predicate": "reviewing_draft",
    "creator_exhausted": "reviewing_draft",
    "creator_final_review": "final_review",
    "creator_edit_conflict": "final_review",
}


@dataclass
class PendingDecision:
    decision_type: str
    payload: dict
    context: dict = field(default_factory=dict)


@dataclass
class CreatorSession:
    session_id: str
    user_input: str
    stage: str = "resolving_entities"
    resolved_entities: dict = field(default_factory=dict)
    lower: int | None = None
    upper: int | None = None
    # Categories the user's per-category checkboxes allow Creator to invent
    # *new* entities in (Phase 10 patch 22, B) — empty by default, matching
    # the checkboxes' own off-by-default. Existing location/artifact/faction
    # entities remain referenceable regardless of this set; it only gates
    # fabricating brand-new ones.
    allowed_new_categories: set = field(default_factory=set)
    draft: object = None  # creator.NarrativeDraft, once composed
    attempts: int = 0
    pending_decision: "PendingDecision | None" = None
    result: dict = field(default_factory=dict)
    _generator: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Entity resolution — tagged entities only, never invents new ones
# ---------------------------------------------------------------------------

def _resolve_creator_tag_gen(tag: str):
    """Disambiguation reuses pipeline_session's own "entity_candidates"
    decision shape, always with allow_create=False — a tag with zero
    matches means this whole request can't proceed (returns None), it
    never falls through to entity creation the way the normal chat
    pipeline's tag resolution does."""
    exact_matches, partial_matches = mapping.find_existing_matches_any_category(tag)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return (
            yield PendingDecision(
                "entity_candidates",
                {"tag": tag, "candidates": exact_matches, "allow_create": False},
                {"tag": tag},
            )
        )
    if partial_matches:
        return (
            yield PendingDecision(
                "entity_candidates",
                {"tag": tag, "candidates": partial_matches, "allow_create": False},
                {"tag": tag},
            )
        )
    return None


# ---------------------------------------------------------------------------
# Relational predicate confirmation — patch 16's registry gate, adapted for
# a flat list of DraftEvents. Run once, on the final Inspector-approved
# draft, not on every retried/discarded attempt.
# ---------------------------------------------------------------------------

def _resolve_relational_predicates_gen(events: list):
    known_relational = {e["id"] for e in schema.load_status_effects() if e.get("type") == "relational"}
    kept = []
    for event in events:
        effect = event.duration_effect
        if not (
            event.event_type == "duration"
            and effect
            and effect.get("target")
            and effect.get("predicate") not in known_relational
        ):
            kept.append(event)
            continue

        proposed = effect["predicate"]
        response = yield PendingDecision(
            "new_relational_predicate",
            {
                "predicate": proposed,
                "entity_id": effect.get("entity"),
                "target_id": effect.get("target"),
                "reason": event.notes,
            },
            {},
        )
        action = (response or {}).get("action") if isinstance(response, dict) else None
        if action in (None, "cancel"):
            continue  # this event is dropped; the rest of the draft still saves
        final_name = (response.get("name") or "").strip() or proposed
        schema.add_status_effect(final_name, final_name, "relational")
        known_relational.add(final_name)
        effect["predicate"] = final_name
        kept.append(event)
    return kept


# ---------------------------------------------------------------------------
# Save — reuses archivist.build_diff exactly (spec F), once per drafted
# event rather than once for the whole batch.
# ---------------------------------------------------------------------------

def _save_draft(resolved_entities: dict, draft) -> list:
    """build_diff assumes every record in one call shares a single
    parsed_input.years[0]/raw_text — true for the normal chat pipeline (one
    sentence -> possibly several records sharing its year) but not for
    Creator's batch, where each event has its own distinct year and prose.
    Applying sequentially rather than computing every event's diff before
    writing any of them is what lets a later event's diff see an earlier
    event's pointer already merged into storage (get_entity/
    get_events_for_entity read live state), instead of needing build_diff's
    own same-call pointer-merging (existing_ids/pointer_targets) to span
    across separate calls, which it was never built to do."""
    applied = []

    # New entities (Phase 10 patch 22, B) are created first, with their own
    # real fields (name, notes, any required enum) — storage.save_entity is
    # a genuine upsert, so the event loop below would eventually create a
    # bare row via pointer registration alone if this step were skipped,
    # but that row would only ever get event_ids populated, never the
    # entity's actual name/notes/etc.
    for new_entity in draft.new_entities:
        storage.save_entity(new_entity.category, new_entity.entity_id, new_entity.fields)
        if new_entity.fields.get("notes"):
            storage.save_to_chroma(
                new_entity.entity_id, new_entity.fields["notes"], {"category": new_entity.category}
            )
        applied.append(
            archivist.ChangeItem(
                action="create",
                category=new_entity.category,
                entity_id=new_entity.entity_id,
                fields=new_entity.fields,
                body=new_entity.fields.get("notes"),
                reason="Creator가 새로 생성한 조연 엔티티",
            )
        )

    for event in draft.events:
        # A "clear" duration_effect may carry only end_year (start_year
        # belongs to the existing record being closed, not this one) — the
        # fallback matters here specifically because archivist's own clear
        # branch falls back to parsed_input.years[0] when duration_effect
        # itself has no end_year, and a bare None there would silently
        # no-op the clear instead of actually closing the record.
        year = event.year if event.event_type == "point" else (event.start_year or event.end_year)
        parsed_input = parser.ParsedInput(years=[year], tags=[], raw_text=event.notes)
        inferred_event = inference.InferredEvent(
            event_type=event.event_type,
            event_summary=event.notes,
            involved_entities=event.involved_entities,
            duration_effect=event.duration_effect,
        )
        # archivist._pick_location scans resolved_entities.values() for a
        # location-category id — the event's own drafted location (if any)
        # isn't in the session-level resolved_entities (only the originally
        # tagged entities are), so it's added here just for this event's
        # own diff, not merged back into the shared dict.
        event_resolved_entities = resolved_entities
        if event.location:
            event_resolved_entities = {**resolved_entities, "_location": event.location}
        diff = archivist.build_diff(parsed_input, event_resolved_entities, inferred_event)
        if isinstance(diff, archivist.ConfirmationNeeded):
            continue  # shouldn't happen -- Inspector already validated this event's shape
        for item in diff:
            storage.save_entity(item.category, item.entity_id, item.fields)
            if item.body:
                storage.save_to_chroma(item.entity_id, item.body, {"category": item.category})
            applied.append(item)
    return applied


def _draft_event_payload(draft) -> list:
    return [
        {
            "index": i,
            "event_type": e.event_type,
            "notes": e.notes,
            "year": e.year,
            "start_year": e.start_year,
            "end_year": e.end_year,
            "location": e.location,
            "involved_entities": e.involved_entities,
            "duration_effect": e.duration_effect,
        }
        for i, e in enumerate(draft.events)
    ]


def _draft_entity_payload(draft) -> list:
    return [
        {"entity_id": e.entity_id, "category": e.category, "fields": e.fields}
        for e in draft.new_entities
    ]


def _apply_year_edits(draft, edits: dict) -> None:
    """`edits`: {event_index (int or str): {"year": .. } or {"start_year": .., "end_year": ..}}.

    Creator's generated notes almost always spell the year out in the prose
    itself ("2080년, 쟝과 미라는..."), since that's what makes the sentence
    checkable at all. Re-inspection (the caller always re-runs Inspector
    after an edit) reasons over that prose text, not the structured year
    field in isolation — so if only the number is updated and the old year
    is left sitting in the notes, a re-check can silently fail to catch a
    contradiction the new year actually creates (verified: editing year
    2080->2050 without this fix left the notes reading "2080년...", which
    doesn't contradict an established "met in 2079" fact, so the edit's own
    real problem was invisible to the very re-check meant to catch it).
    Only touches the notes when the *old* year literally appears in it —
    never guesses at a rewrite otherwise."""
    for index_key, values in edits.items():
        index = int(index_key)
        if not (0 <= index < len(draft.events)):
            continue
        event = draft.events[index]
        if "year" in values:
            old_year = event.year
            new_year = values["year"]
            event.year = new_year
            if old_year is not None and new_year is not None and str(old_year) in event.notes:
                event.notes = event.notes.replace(str(old_year), str(new_year))
        if event.duration_effect is not None:
            if "start_year" in values:
                old_start = event.duration_effect.get("start_year")
                new_start = values["start_year"]
                event.duration_effect["start_year"] = new_start
                if old_start is not None and new_start is not None and str(old_start) in event.notes:
                    event.notes = event.notes.replace(str(old_start), str(new_start))
            if "end_year" in values:
                old_end = event.duration_effect.get("end_year")
                new_end = values["end_year"]
                event.duration_effect["end_year"] = new_end
                if old_end is not None and new_end is not None and str(old_end) in event.notes:
                    event.notes = event.notes.replace(str(old_end), str(new_end))


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def _draft_and_review_gen(session: CreatorSession, resolved_entities: dict, lower: int, upper: int, supplement):
    first_draft = None
    if lower == upper:
        first_draft = creator.compose_narrative(
            resolved_entities,
            session.user_input,
            lower,
            upper,
            supplement=supplement,
            allowed_new_categories=session.allowed_new_categories,
        )
        if first_draft.natural_event_count > 1:
            choice = yield PendingDecision(
                "creator_count_mismatch",
                {"natural_event_count": first_draft.natural_event_count, "year": lower},
                {},
            )
            action = (choice or {}).get("action") if isinstance(choice, dict) else None
            if action == "widen":
                new_lower, new_upper = choice["lower"], choice["upper"]
                window = creator.compute_year_window(list(resolved_entities.values()))
                if (window.lower is not None and new_upper < window.lower) or (
                    window.upper is not None and new_lower > window.upper
                ):
                    return {
                        "status": "rejected",
                        "stage": "year_window",
                        "message": "새로 지정한 범위가 관련 엔티티가 함께 존재하는 기간과 맞지 않습니다.",
                    }
                lower, upper = new_lower, new_upper
                session.lower, session.upper = lower, upper
                first_draft = None  # discard -- constraint changed, compose fresh
            elif action == "cancel":
                return {"status": "cancelled", "stage": "count_mismatch"}
            # "compress" (or any other truthy response): keep first_draft, continue as-is

    reflection = creator.run_reflection_loop(
        resolved_entities,
        session.user_input,
        lower,
        upper,
        supplement=supplement,
        first_draft=first_draft,
        allowed_new_categories=session.allowed_new_categories,
    )
    session.attempts = reflection.attempts

    if not reflection.approved:
        # Never silently fails (spec E) -- show the last attempt + why,
        # let the human decide manually rather than reporting bare failure.
        choice = yield PendingDecision(
            "creator_exhausted",
            {
                "attempts": reflection.attempts,
                "reason": reflection.last_reason,
                "events": _draft_event_payload(reflection.draft),
                "new_entities": _draft_entity_payload(reflection.draft),
            },
            {},
        )
        action = (choice or {}).get("action") if isinstance(choice, dict) else None
        if action != "keep_anyway":
            return {"status": "cancelled", "stage": "reflection_exhausted", "reason": reflection.last_reason}
        # falls through to final review with the unapproved draft -- same
        # shape as a human overriding a rag_check warning elsewhere in the app

    draft = reflection.draft
    kept_events = yield from _resolve_relational_predicates_gen(draft.events)
    draft.events = kept_events
    session.draft = draft

    return (yield from _final_review_gen(session, resolved_entities, lower, upper))


def _final_review_gen(session: CreatorSession, resolved_entities: dict, lower: int, upper: int):
    while True:
        response = yield PendingDecision(
            "creator_final_review",
            {
                "events": _draft_event_payload(session.draft),
                "new_entities": _draft_entity_payload(session.draft),
            },
            {},
        )
        action = (response or {}).get("action") if isinstance(response, dict) else None

        if action == "redo":
            supplement = (response.get("supplement") or "").strip() or None
            return (yield from _draft_and_review_gen(session, resolved_entities, lower, upper, supplement))

        if action == "save":
            edits = response.get("year_edits") or {}
            if edits:
                _apply_year_edits(session.draft, edits)
                reinspect = creator.inspect_draft(resolved_entities, session.draft)
                if not reinspect.approved:
                    confirm = yield PendingDecision("creator_edit_conflict", {"reason": reinspect.reason}, {})
                    confirm_action = (confirm or {}).get("action") if isinstance(confirm, dict) else None
                    if confirm_action != "save_anyway":
                        continue  # back to review -- let them fix the year again
            applied = _save_draft(resolved_entities, session.draft)
            return {
                "status": "saved",
                "resolved_entities": resolved_entities,
                "draft": session.draft,
                "applied": applied,
            }

        # "cancel" or anything unrecognized
        return {"status": "cancelled", "stage": "final_review"}


def _creator_generator(session: CreatorSession):
    parsed = parser.parse_input(session.user_input)
    tags = parsed.tags
    if not tags:
        return {
            "status": "error",
            "message": "엔티티가 태그되지 않았습니다. Creator는 [ ]로 태그된 기존 엔티티만 사용합니다.",
        }

    resolved_entities = {}
    for tag in tags:
        entity_id = yield from _resolve_creator_tag_gen(tag)
        if entity_id is None:
            return {
                "status": "rejected",
                "stage": "entity_resolution",
                "message": (
                    f'"{tag}"와(과) 일치하는 기존 엔티티를 찾을 수 없습니다. Creator는 새 엔티티를 '
                    f"만들지 않고, 이미 존재하는 엔티티만 사용합니다."
                ),
            }
        resolved_entities[tag] = entity_id
    session.resolved_entities = resolved_entities

    entity_ids = list(resolved_entities.values())
    window = creator.compute_year_window(entity_ids)
    lower_hint, upper_hint = creator.parse_year_hint(session.user_input)

    if lower_hint is not None:
        if (window.lower is not None and upper_hint < window.lower) or (
            window.upper is not None and lower_hint > window.upper
        ):
            return {
                "status": "rejected",
                "stage": "year_window",
                "message": (
                    f"요청하신 연도({lower_hint}~{upper_hint}년)가 관련 엔티티가 함께 존재하는 "
                    f"기간과 맞지 않습니다." + (f" {window.reason}" if window.reason else "")
                ),
            }
        lower, upper = lower_hint, upper_hint
    else:
        if not window.possible:
            return {"status": "rejected", "stage": "year_window", "message": window.reason}
        confirmed = yield PendingDecision(
            "creator_year_confirm", {"lower": window.lower, "upper": window.upper}, {}
        )
        if not isinstance(confirmed, dict) or confirmed.get("action") == "cancel":
            return {"status": "cancelled", "stage": "year_window"}
        lower, upper = confirmed["lower"], confirmed["upper"]

    session.lower, session.upper = lower, upper

    return (yield from _draft_and_review_gen(session, resolved_entities, lower, upper, None))


# ---------------------------------------------------------------------------
# Public entry points — mirrors pipeline_session.py's own shape exactly
# ---------------------------------------------------------------------------

def _advance(session: CreatorSession, send_value=None, is_start: bool = False) -> CreatorSession:
    gen = session._generator
    try:
        pending = next(gen) if is_start else gen.send(send_value)
    except StopIteration as stop:
        result = stop.value or {}
        session.result = result
        session.pending_decision = None
        session.stage = "done" if result.get("status") == "saved" else "aborted"
        return session

    session.pending_decision = pending
    session.stage = _STAGE_BY_DECISION.get(pending.decision_type, session.stage)
    return session


def start_session(user_input: str, allowed_new_categories: set | None = None) -> CreatorSession:
    session = CreatorSession(
        session_id=str(uuid.uuid4()), user_input=user_input,
        allowed_new_categories=set(allowed_new_categories or ()),
    )
    session._generator = _creator_generator(session)
    _SESSIONS[session.session_id] = session
    return _advance(session, is_start=True)


def resume_session(session_id: str, decision_response) -> CreatorSession:
    session = _SESSIONS[session_id]
    if session.pending_decision is None:
        raise ValueError(f"세션 {session_id}에는 응답을 기다리는 결정이 없습니다.")
    return _advance(session, send_value=decision_response)
