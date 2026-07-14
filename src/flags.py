"""Flags — Phase 6.5.

Marks a related record surfaced during a Phase 6 field update as "needs a
closer look later," without blocking or auto-fixing anything. A flag is
completely independent of whatever field update triggered it — actually
fixing the flagged entity happens later, via detail_panel.py (or a future
GUI review panel), not here. `flags` is a plain system table, not one of
schema_registry.yaml's entity categories.
"""

import datetime
import sqlite3
from dataclasses import dataclass

from . import storage


@dataclass
class Flag:
    id: int
    entity_id: str
    flagged_from: str
    reason: str | None
    created_at: str


def _init_flags_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT NOT NULL,
            flagged_from TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        )
        """
    )


def _row_to_flag(row: sqlite3.Row) -> Flag:
    return Flag(
        id=row["id"],
        entity_id=row["entity_id"],
        flagged_from=row["flagged_from"],
        reason=row["reason"],
        created_at=row["created_at"],
    )


def add_flag(entity_id: str, flagged_from: str, reason: str | None = None) -> Flag:
    """Always inserts a new row — the same entity_id may be flagged more
    than once, from different contexts, and each occurrence is kept."""
    conn = storage.get_connection()
    _init_flags_table(conn)

    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    cursor = conn.execute(
        "INSERT INTO flags (entity_id, flagged_from, reason, created_at) VALUES (?, ?, ?, ?)",
        (entity_id, flagged_from, reason, created_at),
    )
    conn.commit()
    flag_id = cursor.lastrowid
    conn.close()

    return Flag(
        id=flag_id,
        entity_id=entity_id,
        flagged_from=flagged_from,
        reason=reason,
        created_at=created_at,
    )


def list_flags() -> list:
    conn = storage.get_connection()
    _init_flags_table(conn)
    rows = conn.execute("SELECT * FROM flags ORDER BY id ASC").fetchall()
    conn.close()
    return [_row_to_flag(row) for row in rows]


def clear_flag(flag_id: int) -> None:
    conn = storage.get_connection()
    _init_flags_table(conn)
    conn.execute("DELETE FROM flags WHERE id = ?", (flag_id,))
    conn.commit()
    conn.close()


def list_flags_deduped() -> list:
    """Same as list_flags(), but collapsed to one row per entity_id — the
    same entity can end up flagged from several different edit contexts, and
    the review list should show it once, not once per occurrence. The most
    recently created flag (highest id, since list_flags() is id ASC) is kept
    as the representative for that entity's reason/timestamp."""
    latest_by_entity = {}
    for flag in list_flags():
        latest_by_entity[flag.entity_id] = flag
    return list(latest_by_entity.values())


def clear_flags_for_entity(entity_id: str) -> int:
    """Delete every unresolved flag for entity_id, regardless of how many
    different contexts (flagged_from) they came from. Returns the number of
    rows deleted, so a caller can report "N flags auto-cleared"."""
    conn = storage.get_connection()
    _init_flags_table(conn)
    cursor = conn.execute("DELETE FROM flags WHERE entity_id = ?", (entity_id,))
    conn.commit()
    cleared = cursor.rowcount
    conn.close()
    return cleared
