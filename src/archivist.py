"""Archivist (Step 5 prep) — Phase 4.

Pure assembly: Phase 3 already ran every LLM-backed inference and RAG
verification, so this only combines those results into a structured diff.
No LLM calls here — that's the point of splitting this out from Phase 3, so
LLM-dependent logic and assembly logic can be debugged independently.

Design note: relationships are an append-only history log (rule 4 — they're
always "create", never "update"). A reversible status's "current" value
therefore can't live on the relationship that originally set it; it lives on
a dedicated `active_status_effects` snapshot field on the entity itself
(character/location/faction/artifact), which this module updates.
"""

import re
from dataclasses import dataclass

from . import schema, storage

_STATUS_BEARING_CATEGORIES = ("character", "location", "faction", "artifact")


@dataclass
class ChangeItem:
    action: str  # "create" | "update"
    category: str
    entity_id: str
    fields: dict
    body: str | None
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


def _next_active_status_effects(
    entity_id: str, category: str, effect_id: str, action: str, year: int
) -> list:
    """Each entry is {"status": id, "start_year": int, "end_year": int|None}
    (Phase 9 status-range patch — was a bare status-id list before). "set"
    opens a new range at `year` unless one's already open; "clear" closes
    the open range for that status at `year` (closing the most recently
    opened one if more than one is somehow open — that shouldn't normally
    happen, but this keeps clear() well-defined either way)."""
    entity = storage.get_entity(category, entity_id) or {}
    current = [dict(r) for r in (entity.get("active_status_effects") or [])]

    if action == "set":
        already_open = any(
            r["status"] == effect_id and r.get("end_year") is None for r in current
        )
        if not already_open:
            current.append({"status": effect_id, "start_year": year, "end_year": None})
    elif action == "clear":
        open_ranges = [
            r for r in current if r["status"] == effect_id and r.get("end_year") is None
        ]
        if open_ranges:
            target = max(open_ranges, key=lambda r: r["start_year"])
            target["end_year"] = year

    return current


def build_diff(
    parsed_input, resolved_entities: dict, inferred_event, rag_judgments: list
) -> list:
    changes = []
    existing_ids = set()

    location_id = next(
        (
            eid
            for eid in resolved_entities.values()
            if schema.category_from_id(eid) == "location"
        ),
        None,
    )

    status_effect = inferred_event.status_effect or {}
    status_action = status_effect.get("action")
    status_entity = status_effect.get("entity")
    status_effect_id = status_effect.get("effect")

    timeline_status_value = status_effect_id if status_action == "set" else None

    timeline_reason = f"새 사건 기록: {inferred_event.event_summary}"
    if rag_judgments:
        notes = "; ".join(f"[{j.type}] {j.reason}" for j in rag_judgments)
        timeline_reason += f" (RAG 검증 메모: {notes})"

    timeline_id = generate_id("timeline", inferred_event.event_summary, existing_ids)
    existing_ids.add(timeline_id)
    changes.append(
        ChangeItem(
            action="create",
            category="timeline",
            entity_id=timeline_id,
            fields={
                "year": parsed_input.year,
                "location": location_id,
                "status_effect": timeline_status_value,
                "notes": inferred_event.event_summary,
            },
            body=parsed_input.raw_text,
            reason=timeline_reason,
        )
    )

    for rel in inferred_event.relationships:
        subject = rel.get("subject")
        predicate = rel.get("predicate")
        obj = rel.get("object")

        rel_id = generate_id(
            "relationship", f"{subject}_{predicate}_{obj}", existing_ids
        )
        existing_ids.add(rel_id)
        changes.append(
            ChangeItem(
                action="create",
                category="relationship",
                entity_id=rel_id,
                fields={
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "since": parsed_input.year,
                    "until": None,
                    "notes": inferred_event.event_summary,
                },
                body=parsed_input.raw_text,
                reason=f"{subject}와(과) {obj} 사이의 '{predicate}' 관계 기록",
            )
        )

    if status_action in ("set", "clear") and status_entity and status_effect_id:
        status_category = schema.category_from_id(status_entity)
        if status_category in _STATUS_BEARING_CATEGORIES:
            new_effects = _next_active_status_effects(
                status_entity, status_category, status_effect_id, status_action,
                parsed_input.year,
            )
            verb = "해제" if status_action == "clear" else "부여"
            changes.append(
                ChangeItem(
                    action="update",
                    category=status_category,
                    entity_id=status_entity,
                    fields={"active_status_effects": new_effects},
                    body=None,
                    reason=f"{status_entity}의 '{status_effect_id}' 상태가 이 사건으로 {verb}됨",
                )
            )

    return changes
