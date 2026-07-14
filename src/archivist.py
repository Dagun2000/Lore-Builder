"""Archivist (Step 5 prep) — Phase 10 rewrite.

Pure assembly: inference.py already ran the LLM-backed judgment, so this
only turns its InferredEvent into a structured diff. No LLM calls here.

Phase 10's event-centric redesign collapses what used to be "one timeline
create + N relationship creates + maybe one status update" into a single
timeline record (point or duration) plus event_ids pointer updates on
whichever entities are involved — no separate relationship category, no
`active_status_effects` snapshot field. Reading current storage state here
(to compute the next event_ids list, or find the open duration record a
"clear" should close) is the same established pattern the old
_next_active_status_effects used — archivist reads current state to compute
a diff, it just never writes anything itself.
"""

import re
from dataclasses import dataclass

from . import schema, storage

_EVENT_POINTER_CATEGORIES = ("character", "location", "faction", "artifact", "race")


@dataclass
class ChangeItem:
    action: str  # "create" | "update"
    category: str
    entity_id: str
    fields: dict
    body: str | None
    reason: str


@dataclass
class ConfirmationNeeded:
    """Returned by build_diff instead of a diff list when the input can't be
    turned into one record — either inference.py judged the sentence
    ambiguous (is_single_event False), or a "clear" action named a status/
    relationship that isn't actually open. Phase 8's state machine surfaces
    this as a "multi_event_warning" pending_decision with [계속 진행]/[취소]."""

    reason: str


def _slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w]", "", text)
    return text[:60] or "entry"


def generate_id(category: str, seed_text: str, existing_ids: set) -> str:
    prefix = schema.load_schema_registry()[category]["id_prefix"]
    candidate = f"{prefix}{_slugify(seed_text)}"

    entity_id = candidate
    suffix = 2
    while entity_id in existing_ids or storage.entity_exists(category, entity_id):
        entity_id = f"{candidate}_{suffix}"
        suffix += 1
    return entity_id


def _pick_location(resolved_entities: dict) -> str | None:
    return next(
        (eid for eid in resolved_entities.values() if schema.category_from_id(eid) == "location"),
        None,
    )


def _next_event_ids(entity_id: str, event_id: str) -> list:
    category = schema.category_from_id(entity_id)
    if category is None:
        return [event_id]
    entity = storage.get_entity(category, entity_id) or {}
    current = list(entity.get("event_ids") or [])
    if event_id not in current:
        current.append(event_id)
    return current


def _event_pointer_change(entity_id: str, event_id: str) -> ChangeItem | None:
    category = schema.category_from_id(entity_id)
    if category is None or category not in _EVENT_POINTER_CATEGORIES:
        return None
    return ChangeItem(
        action="update",
        category=category,
        entity_id=entity_id,
        fields={"event_ids": _next_event_ids(entity_id, event_id)},
        body=None,
        reason=f"{entity_id}의 이벤트 기록에 {event_id} 추가",
    )


def _build_point_diff(parsed_input, resolved_entities: dict, inferred_event) -> list:
    changes = []
    existing_ids = set()

    timeline_id = generate_id("timeline", inferred_event.event_summary, existing_ids)
    existing_ids.add(timeline_id)
    year = parsed_input.years[0]

    changes.append(
        ChangeItem(
            action="create",
            category="timeline",
            entity_id=timeline_id,
            fields={
                "year": year,
                "location": _pick_location(resolved_entities),
                "notes": inferred_event.event_summary,
            },
            body=parsed_input.raw_text,
            reason=f"새 사건 기록: {inferred_event.event_summary}",
        )
    )

    involved = inferred_event.involved_entities or list(resolved_entities.values())
    for entity_id in involved:
        change = _event_pointer_change(entity_id, timeline_id)
        if change is not None:
            changes.append(change)

    return changes


def _build_duration_diff(parsed_input, inferred_event):
    effect = inferred_event.duration_effect or {}
    entity_id = effect.get("entity")
    predicate = effect.get("predicate")
    target = effect.get("target")
    action = effect.get("action")

    if not entity_id or not predicate or action not in ("set", "clear", "set_closed"):
        return ConfirmationNeeded(
            reason="상태/관계 판단 결과가 불완전해 기록을 만들 수 없습니다. 입력을 다시 확인해주세요."
        )

    changes = []
    existing_ids = set()

    if action in ("set", "set_closed"):
        timeline_id = generate_id(
            "timeline", f"{entity_id}_{predicate}_{target or ''}", existing_ids
        )
        existing_ids.add(timeline_id)
        summary = inferred_event.event_summary or f"{entity_id}의 '{predicate}' 상태/관계"
        changes.append(
            ChangeItem(
                action="create",
                category="timeline",
                entity_id=timeline_id,
                fields={
                    "entity": entity_id,
                    "predicate": predicate,
                    "target": target,
                    "start_year": effect.get("start_year"),
                    "end_year": effect.get("end_year"),
                    "notes": summary,
                },
                body=parsed_input.raw_text,
                reason=f"{entity_id}의 '{predicate}' 상태/관계 시작 기록",
            )
        )
        for participant in ([entity_id, target] if target else [entity_id]):
            change = _event_pointer_change(participant, timeline_id)
            if change is not None:
                changes.append(change)

        if predicate == "owns" and target and schema.category_from_id(target) == "artifact":
            changes.append(
                ChangeItem(
                    action="update",
                    category="artifact",
                    entity_id=target,
                    fields={"current_owner": entity_id},
                    body=None,
                    reason=f"{target}의 current_owner 캐시를 최신 소유 기록으로 갱신",
                )
            )

    else:  # clear
        open_records = [
            r
            for r in storage.get_duration_records(entity_id, predicate)
            if r.get("entity") == entity_id and r.get("end_year") is None
        ]
        if not open_records:
            return ConfirmationNeeded(
                reason=f"해제할 대상 상태를 찾지 못했습니다: {entity_id}의 '{predicate}' 상태/관계가 열려있지 않습니다."
            )
        target_record = max(open_records, key=lambda r: r.get("start_year") or 0)
        end_year = effect.get("end_year") or parsed_input.years[0]
        changes.append(
            ChangeItem(
                action="update",
                category="timeline",
                entity_id=target_record["id"],
                fields={"end_year": end_year},
                body=None,
                reason=f"{entity_id}의 '{predicate}' 상태/관계 해제 (end_year={end_year})",
            )
        )

    return changes


def build_diff(parsed_input, resolved_entities: dict, inferred_event):
    if not inferred_event.is_single_event:
        return ConfirmationNeeded(reason=inferred_event.ambiguity_reason)

    if inferred_event.event_type == "point":
        return _build_point_diff(parsed_input, resolved_entities, inferred_event)

    return _build_duration_diff(parsed_input, inferred_event)
