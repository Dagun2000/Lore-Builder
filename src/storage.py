"""Storage layer shell for Lore Builder.

Phase 0 scope only: dynamic table creation from the schema registry and
plain CRUD against SQLite, plus a thin wrapper around a local persistent
Chroma collection. No validation logic lives here.
"""

import json
import sqlite3
from pathlib import Path

import chromadb

from .schema import category_from_id, load_schema_registry

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "lore.db"
CHROMA_PATH = BASE_DIR / "chroma_store"

_FIELD_TYPE_TO_SQL = {
    "integer": "INTEGER",
    "text": "TEXT",
    "boolean": "INTEGER",
    "enum": "TEXT",
    "reference": "TEXT",
    "list": "TEXT",  # stored as JSON
}

_SERIALIZE_TYPES = {"list"}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _columns_for(category: str) -> list:
    registry = load_schema_registry()
    fields = registry[category]["fields"]
    return [f["name"] for f in fields]


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create one table per schema-registry category if it doesn't exist yet."""
    owns_conn = conn is None
    conn = conn or get_connection()
    registry = load_schema_registry()
    for category, definition in registry.items():
        columns = ", ".join(
            f'"{f["name"]}" {_FIELD_TYPE_TO_SQL.get(f["type"], "TEXT")}'
            for f in definition["fields"]
        )
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{category}" ('
            f'id TEXT PRIMARY KEY{"," if columns else ""} {columns}'
            f')'
        )
    conn.commit()
    if owns_conn:
        conn.close()


def _serialize(field_type: str, value):
    if value is not None and field_type in _SERIALIZE_TYPES:
        return json.dumps(value)
    return value


def _deserialize(field_type: str, value):
    if value is not None and field_type in _SERIALIZE_TYPES:
        return json.loads(value)
    return value


def save_entity(category: str, entity_id: str, fields: dict) -> None:
    """Upsert an entity: merges `fields` on top of any existing row."""
    conn = get_connection()
    init_db(conn)

    registry = load_schema_registry()
    field_defs = {f["name"]: f for f in registry[category]["fields"]}

    existing = get_entity(category, entity_id, _conn=conn) or {}
    merged = {**existing, **fields}

    columns = list(field_defs.keys())
    values = [
        _serialize(field_defs[col]["type"], merged.get(col))
        for col in columns
    ]

    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(f'"{c}"' for c in columns)
    conn.execute(
        f'INSERT OR REPLACE INTO "{category}" (id, {column_list}) '
        f'VALUES (?, {placeholders})',
        [entity_id] + values,
    )
    conn.commit()
    conn.close()


def get_entity(category: str, entity_id: str, _conn: sqlite3.Connection | None = None) -> dict | None:
    owns_conn = _conn is None
    conn = _conn or get_connection()
    init_db(conn)

    registry = load_schema_registry()
    field_defs = {f["name"]: f for f in registry[category]["fields"]}

    row = conn.execute(
        f'SELECT * FROM "{category}" WHERE id = ?', (entity_id,)
    ).fetchone()

    if owns_conn:
        conn.close()

    if row is None:
        return None

    result = {"id": row["id"]}
    for name, definition in field_defs.items():
        result[name] = _deserialize(definition["type"], row[name])
    return result


def entity_exists(category: str, entity_id: str) -> bool:
    return get_entity(category, entity_id) is not None


def list_entities(category: str) -> list:
    """Every row in `category`'s table, deserialized like get_entity() —
    used by the GUI's dictionary/search/reference-picker views, which need
    the whole table rather than one row at a time."""
    conn = get_connection()
    init_db(conn)

    registry = load_schema_registry()
    field_defs = {f["name"]: f for f in registry[category]["fields"]}

    rows = conn.execute(f'SELECT * FROM "{category}"').fetchall()
    conn.close()

    results = []
    for row in rows:
        result = {"id": row["id"]}
        for name, definition in field_defs.items():
            result[name] = _deserialize(definition["type"], row[name])
        results.append(result)
    return results


def get_events_for_entity(entity_id: str) -> list:
    """Every timeline record (point + duration, mixed) pointed at by
    entity_id.event_ids, year-sorted — point events by `year`, duration
    events by `start_year`. This is the single source of truth for "what's
    related to this entity" (Phase 10's event-centric redesign): no separate
    relationship table, no vector search, just the pointer list."""
    category = category_from_id(entity_id)
    if category is None:
        return []

    entity = get_entity(category, entity_id)
    if entity is None:
        return []

    records = []
    for event_id in entity.get("event_ids") or []:
        record = get_entity("timeline", event_id)
        if record is not None:
            records.append(record)

    records.sort(key=lambda r: r["year"] if r.get("year") is not None else (r.get("start_year") or 0))
    return records


def get_event_years(entity_id: str) -> list:
    """Every year entity_id is on record for — a point event's `year`, or a
    duration event's `start_year`/`end_year` (both count: hard_check's
    terminal/lifespan checks care about the full span an entity is attested
    across, not just when a status began)."""
    years = set()
    for record in get_events_for_entity(entity_id):
        if record.get("year") is not None:
            years.add(record["year"])
        if record.get("start_year") is not None:
            years.add(record["start_year"])
        if record.get("end_year") is not None:
            years.add(record["end_year"])
    return sorted(years)


def get_duration_records(entity_id: str, predicate: str | None = None) -> list:
    """Duration-event records where entity_id is the `entity` or the
    `target` side, optionally filtered to one predicate. Sourced from
    entity_id.event_ids (not a raw table scan) so this never drifts from
    what get_events_for_entity/get_event_years already see."""
    records = [
        r
        for r in get_events_for_entity(entity_id)
        if r.get("start_year") is not None
        and (r.get("entity") == entity_id or r.get("target") == entity_id)
    ]
    if predicate is not None:
        records = [r for r in records if r.get("predicate") == predicate]
    return records


def get_current_state(entity_id: str, predicate: str, year: int | None = None) -> list:
    """Targets of entity_id's `predicate` duration records that are active
    at `year` (or, if `year` is omitted, still open — end_year is None).
    For a personal status (no target, e.g. predicate="imprisoned"), the
    returned list still holds one `None` entry per active record — check
    truthiness (`if get_current_state(...):`) to answer "is this active"."""
    active_targets = []
    for record in get_duration_records(entity_id, predicate):
        if record.get("entity") != entity_id:
            continue
        start = record["start_year"]
        end = record.get("end_year")
        is_active = (end is None) if year is None else (start <= year and (end is None or year <= end))
        if is_active:
            active_targets.append(record.get("target"))
    return active_targets


def get_current_holder(target_id: str, predicate: str = "owns", year: int | None = None) -> str | None:
    """Reverse of get_current_state: which entity currently has `predicate`
    pointed *at* target_id (e.g. who currently owns this artifact), computed
    from duration records on read. Replaces artifact.current_owner (Phase 10
    patch 7 follow-up) — there is no stored "current" field/cache anywhere
    for this, on the same principle that removed artifact.current_status:
    current state is a query, not a column. Returns None if nothing's open
    (or, if `year` is given, nothing covers that year)."""
    candidates = [
        r for r in get_duration_records(target_id, predicate) if r.get("target") == target_id
    ]
    active = []
    for record in candidates:
        start = record.get("start_year")
        if start is None:
            continue
        end = record.get("end_year")
        is_active = (end is None) if year is None else (start <= year and (end is None or year <= end))
        if is_active:
            active.append((start, record.get("entity")))
    if not active:
        return None
    # If somehow more than one is open at once, the most recently started
    # one wins — matches how a "set" duration event is meant to supersede
    # whatever came before it.
    return max(active, key=lambda pair: pair[0])[1]


def add_event_pointer(entity_id: str, event_id: str) -> None:
    """Add event_id to entity_id.event_ids, deduped."""
    category = category_from_id(entity_id)
    if category is None:
        return
    entity = get_entity(category, entity_id) or {}
    current = list(entity.get("event_ids") or [])
    if event_id not in current:
        current.append(event_id)
        save_entity(category, entity_id, {"event_ids": current})


def remove_event_pointer(entity_id: str, event_id: str) -> None:
    """Remove event_id from entity_id.event_ids, if present."""
    category = category_from_id(entity_id)
    if category is None:
        return
    entity = get_entity(category, entity_id) or {}
    current = list(entity.get("event_ids") or [])
    if event_id in current:
        save_entity(category, entity_id, {"event_ids": [e for e in current if e != event_id]})


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _chroma_client


def get_chroma_collection():
    return get_chroma_client().get_or_create_collection("lore")


def save_to_chroma(entity_id: str, text_body: str, metadata: dict) -> None:
    collection = get_chroma_collection()
    collection.upsert(ids=[entity_id], documents=[text_body], metadatas=[metadata])


def save_to_chroma_batch(items: list) -> None:
    """Upsert several documents in one Chroma call instead of one call per
    document — each item in `items` is (entity_id, text_body, metadata).

    Measured, not assumed: saving N documents via N separate save_to_chroma
    calls (each computes its own embedding locally, ~0.3s+ per call — see
    save_to_chroma) scales linearly with N and dominates how long a
    multi-event Creator save feels; batching the same N documents into one
    upsert call measured ~3.5x faster (embedding computation amortizes far
    better in bulk than repeated per-call overhead), and is otherwise
    identical in effect. No-op on an empty list."""
    if not items:
        return
    ids, documents, metadatas = zip(*items)
    get_chroma_collection().upsert(ids=list(ids), documents=list(documents), metadatas=list(metadatas))


def query_chroma(query_text: str, top_k: int = 3, ids: list | None = None) -> dict:
    """`ids`, when given, restricts the similarity search to that subset of
    documents instead of the whole collection — used by Phase 6's
    find_related_context to *rank* a fixed candidate pool rather than
    open a fresh whole-collection search.

    Not every id in `ids` is guaranteed to have a Chroma document — SQLite
    and Chroma are two separate stores, and anything saved via save_entity
    without a matching save_to_chroma call (a test fixture, a hand-edited
    row, an older entity from before this code path existed) leaves a gap.
    collection.query(ids=...) hard-errors on an id it can't find, where
    collection.get(ids=...) just silently returns whatever subset exists —
    so check existence with .get() first and drop anything missing, rather
    than let the whole ranking crash over one orphaned id."""
    collection = get_chroma_collection()
    kwargs = {"query_texts": [query_text], "n_results": top_k}
    if ids:
        existing = set(collection.get(ids=ids)["ids"])
        filtered_ids = [i for i in ids if i in existing]
        if not filtered_ids:
            return {"ids": [[]], "documents": [[]]}
        kwargs["ids"] = filtered_ids
        kwargs["n_results"] = min(top_k, len(filtered_ids))
    return collection.query(**kwargs)


def delete_from_chroma(entity_id: str) -> None:
    collection = get_chroma_collection()
    existing = set(collection.get(ids=[entity_id])["ids"])
    if entity_id in existing:
        collection.delete(ids=[entity_id])


# ---------------------------------------------------------------------------
# Deletion (Phase 10) — SQLite row + Chroma doc, always together
# ---------------------------------------------------------------------------

def delete_row(category: str, entity_id: str) -> None:
    conn = get_connection()
    init_db(conn)
    conn.execute(f'DELETE FROM "{category}" WHERE id = ?', (entity_id,))
    conn.commit()
    conn.close()


def delete_entity_everywhere(category: str, entity_id: str) -> None:
    """Remove entity_id's row from SQLite and its document from Chroma —
    the two stores are always deleted together, never one without the
    other (mirrors save_entity/save_to_chroma always being called as a
    pair when something is created)."""
    delete_row(category, entity_id)
    delete_from_chroma(entity_id)


_EVENT_POINTER_CATEGORIES = ("character", "location", "faction", "artifact", "race")


def find_entities_referencing_event(event_id: str) -> list:
    """[(category, entity_id), ...] for every entity whose event_ids
    contains event_id — a plain scan across the categories that carry
    event_ids at all, rather than a separate reverse-index table; there are
    only a handful of entities per category in practice, so this is cheap."""
    matches = []
    for category in _EVENT_POINTER_CATEGORIES:
        for entity in list_entities(category):
            if event_id in (entity.get("event_ids") or []):
                matches.append((category, entity["id"]))
    return matches
