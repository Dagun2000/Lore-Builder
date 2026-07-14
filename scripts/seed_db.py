"""Seed SQLite + Chroma from the markdown files under db/{category}/.

Each markdown file has a YAML frontmatter block (id + schema fields)
followed by free-text body content used as the Chroma document.
"""

import sys
from pathlib import Path

# Windows consoles default stdout to the active code page (cp949 on Korean
# Windows), not UTF-8 — the "seeded character/쟝" style progress lines would
# otherwise come out as mojibake regardless of how correctly the seed files
# themselves are encoded.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import yaml

from src.schema import load_schema_registry
from src.storage import init_db, save_entity, save_to_chroma

DB_DIR = BASE_DIR / "db"


def parse_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    _, frontmatter_raw, body = text.split("---", 2)
    frontmatter = yaml.safe_load(frontmatter_raw) or {}
    return frontmatter, body.strip()


def sanitize_metadata(category: str, frontmatter: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool (no None/list)."""
    metadata = {"category": category}
    for key, value in frontmatter.items():
        if key == "id" or value is None:
            continue
        if isinstance(value, list):
            metadata[key] = ", ".join(str(v) for v in value)
        else:
            metadata[key] = value
    return metadata


def seed() -> int:
    init_db()
    registry = load_schema_registry()
    count = 0

    for category in registry:
        category_dir = DB_DIR / category
        if not category_dir.is_dir():
            continue

        for md_file in sorted(category_dir.glob("*.md")):
            frontmatter, body = parse_markdown(md_file)
            entity_id = frontmatter.get("id", md_file.stem)
            fields = {k: v for k, v in frontmatter.items() if k != "id"}

            save_entity(category, entity_id, fields)
            save_to_chroma(entity_id, body, sanitize_metadata(category, frontmatter))
            count += 1
            print(f"seeded {category}/{entity_id}")

    return count


if __name__ == "__main__":
    total = seed()
    print(f"Done. Seeded {total} entities.")
