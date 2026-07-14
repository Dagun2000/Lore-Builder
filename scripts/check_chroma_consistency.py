"""Audit: every entity in SQLite should have a matching Chroma document.

A gap here is what crashes field_update.find_related_context (Chroma's
collection.query(ids=...) errors on an unknown id) — storage.query_chroma
was hardened to tolerate gaps, but this script exists to actually find and,
optionally, backfill them so gaps don't accumulate unnoticed.

Usage:
    python scripts/check_chroma_consistency.py            # report only
    python scripts/check_chroma_consistency.py --backfill # also fix gaps
"""

import sys
from pathlib import Path

# Windows consoles default stdout to the active code page (cp949 on Korean
# Windows), not UTF-8 — see main.py's identical block.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.schema import list_categories
from src.storage import get_chroma_collection, list_entities, save_to_chroma


def check(backfill: bool = False) -> list:
    collection = get_chroma_collection()
    missing = []

    for category in list_categories():
        entities = list_entities(category)
        if not entities:
            continue
        ids = [e["id"] for e in entities]
        existing = set(collection.get(ids=ids)["ids"])
        for entity in entities:
            if entity["id"] in existing:
                continue
            missing.append((category, entity))

    for category, entity in missing:
        print(f"missing chroma doc: {category}/{entity['id']}")
        if backfill:
            body = entity.get("notes") or entity.get("name") or entity["id"]
            save_to_chroma(entity["id"], body, {"category": category})
            print(f"  -> backfilled from {'notes' if entity.get('notes') else 'name/id'}")

    if not missing:
        print("No gaps found — every SQLite entity has a matching Chroma document.")
    else:
        print(f"\n{len(missing)} gap(s) found" + (", backfilled." if backfill else " (re-run with --backfill to fix)."))

    return missing


if __name__ == "__main__":
    check(backfill="--backfill" in sys.argv)
