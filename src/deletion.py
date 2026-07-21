"""Deletion — Phase 10.

Event-centric storage makes deletion a pointer-cleanup problem instead of a
cascading-foreign-key problem: deleting an event just means dropping it from
whoever's event_ids list contained it; deleting an entity just means walking
its own event_ids and deciding, per event, whether anyone else still needs
that record.
"""

from dataclasses import dataclass, field

from . import flags, schema, storage


@dataclass
class DeletionResult:
    deleted_id: str
    affected_entities: list = field(default_factory=list)  # entities whose event_ids changed
    deleted_events: list = field(default_factory=list)  # events cascade-deleted (entity deletion only)


def delete_event(event_id: str) -> DeletionResult:
    """Remove event_id from every entity that points at it, then delete the
    timeline record itself from SQLite + Chroma. No entity is left with a
    dangling pointer — that's the whole point of tracking references via
    event_ids instead of a separate join table.

    Also clears any flag sitting on event_id itself (flags.py has its own
    id-keyed table, entirely separate from event_ids pointers) — flagging
    something for a later look and then deleting the very thing flagged
    used to leave the flag dangling forever, pointing at an entity_id
    nothing resolves to anymore, since nothing in this module touched
    flags at all before. Every *edit* path (app.py's field-save and
    event-save buttons) already called flags.clear_flags_for_entity on
    success; deletion just never got the same treatment."""
    referencing = storage.find_entities_referencing_event(event_id)
    for _category, entity_id in referencing:
        storage.remove_event_pointer(entity_id, event_id)

    storage.delete_entity_everywhere("timeline", event_id)
    flags.clear_flags_for_entity(event_id)

    return DeletionResult(
        deleted_id=event_id,
        affected_entities=[entity_id for _category, entity_id in referencing],
    )


def request_entity_deletion(entity_id: str) -> list:
    """Every event entity_id is involved in, full content included — shown
    to the user before delete_entity is actually called, so they can back
    out and fix/keep something instead."""
    return storage.get_events_for_entity(entity_id)


def delete_entity(entity_id: str, category: str) -> DeletionResult:
    """Call after the user has reviewed request_entity_deletion's output and
    chosen to proceed anyway. For each event entity_id is involved in: if
    entity_id was the only one referencing it, the event is now meaningless
    on its own, so cascade-delete it via delete_event; if someone else still
    references it, leave the event alone and just drop entity_id's own
    pointer. Then delete entity_id itself."""
    events = storage.get_events_for_entity(entity_id)
    deleted_events = []
    affected_entities = set()

    for record in events:
        event_id = record["id"]
        referencing = storage.find_entities_referencing_event(event_id)
        other_participants = [eid for _category, eid in referencing if eid != entity_id]

        if other_participants:
            storage.remove_event_pointer(entity_id, event_id)
        else:
            result = delete_event(event_id)
            deleted_events.append(event_id)
            affected_entities.update(result.affected_entities)

    storage.delete_entity_everywhere(category, entity_id)
    flags.clear_flags_for_entity(entity_id)
    affected_entities.discard(entity_id)

    return DeletionResult(
        deleted_id=entity_id,
        affected_entities=sorted(affected_entities),
        deleted_events=deleted_events,
    )
