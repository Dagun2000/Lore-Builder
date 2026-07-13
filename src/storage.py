"""Storage layer shell for the lore reviewer.

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


def find_related_timeline_ids(entity_id: str) -> list:
    """Timeline ids related to entity_id, via:
    - timeline.location referencing entity_id directly
    - relationship rows where entity_id is subject/object and the other
      side is a timeline event (event_ prefix)
    Shared traversal used by get_event_years (Phase 0) and
    field_update.find_related_context (Phase 6) — extend here, not in
    either caller, so the two never drift apart."""
    conn = get_connection()
    init_db(conn)

    seen = set()
    ids = []

    for row in conn.execute(
        'SELECT id FROM "timeline" WHERE location = ?', (entity_id,)
    ):
        if row["id"] not in seen:
            seen.add(row["id"])
            ids.append(row["id"])

    for row in conn.execute(
        'SELECT subject, object FROM "relationship" WHERE subject = ? OR object = ?',
        (entity_id, entity_id),
    ):
        other = row["object"] if row["subject"] == entity_id else row["subject"]
        if other and other.startswith("event_") and other not in seen:
            event_row = conn.execute(
                'SELECT id FROM "timeline" WHERE id = ?', (other,)
            ).fetchone()
            if event_row:
                seen.add(event_row["id"])
                ids.append(event_row["id"])

    conn.close()
    return ids


def get_event_years(entity_id: str) -> list:
    """Years of every timeline record related to entity_id (see
    find_related_timeline_ids for how "related" is determined)."""
    ids = find_related_timeline_ids(entity_id)
    if not ids:
        return []

    conn = get_connection()
    init_db(conn)

    placeholders = ", ".join("?" for _ in ids)
    years = {
        row["year"]
        for row in conn.execute(
            f'SELECT year FROM "timeline" WHERE id IN ({placeholders})', ids
        )
        if row["year"] is not None
    }

    conn.close()
    return sorted(years)


def get_status_effects(entity_id: str) -> list:
    """Current active status_effect ids for entity_id.

    This reads the entity's `active_status_effects` snapshot field — the
    single source of truth for "what status is active now", kept in sync by
    archivist.build_diff's update ChangeItems (Phase 4). Do not reintroduce a
    timeline/relationship history scan here: relationships are an
    append-only log with no "resolved" flag, so a scan can never tell a
    cleared status from an active one and would drift from what was
    actually saved."""
    category = category_from_id(entity_id)
    if category is None:
        return []

    entity = get_entity(category, entity_id)
    if entity is None:
        return []

    return list(entity.get("active_status_effects") or [])


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


def query_chroma(query_text: str, top_k: int = 3, ids: list | None = None) -> dict:
    """`ids`, when given, restricts the similarity search to that subset of
    documents instead of the whole collection — used by Phase 6's
    find_related_context to *rank* a fixed candidate pool rather than
    open a fresh whole-collection search."""
    collection = get_chroma_collection()
    kwargs = {"query_texts": [query_text], "n_results": top_k}
    if ids:
        kwargs["ids"] = ids
    return collection.query(**kwargs)
