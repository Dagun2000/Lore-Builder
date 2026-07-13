"""Schema registry loader for the lore reviewer.

Parses schema_registry.yaml and status_effects.yaml into plain
Python dict/list structures and exposes small lookup helpers on top.
"""

from pathlib import Path
from functools import lru_cache

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_REGISTRY_PATH = BASE_DIR / "schema_registry.yaml"
STATUS_EFFECTS_PATH = BASE_DIR / "status_effects.yaml"


@lru_cache(maxsize=1)
def load_schema_registry() -> dict:
    """Return the parsed schema_registry.yaml as {category: {id_prefix, fields}}."""
    with open(SCHEMA_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_status_effects() -> list:
    """Return the parsed status_effects.yaml as a list of {id, label} dicts."""
    with open(STATUS_EFFECTS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["status_effects"]


def get_category_schema(category: str) -> dict:
    registry = load_schema_registry()
    if category not in registry:
        raise KeyError(f"Unknown category: {category}")
    return registry[category]


def get_fields(category: str) -> list:
    """Return the raw list of field definitions for a category."""
    return get_category_schema(category)["fields"]


def get_fields_with_role(category: str, role: str) -> list:
    """Return field definitions in `category` whose `role` matches."""
    return [f for f in get_fields(category) if f.get("role") == role]


def get_required_fields(category: str) -> list:
    """Return field definitions in `category` where required is true."""
    return [f for f in get_fields(category) if f.get("required")]


def category_from_id(entity_id: str) -> str | None:
    """Reverse-match an entity id (e.g. char_jang) to its category via id_prefix."""
    registry = load_schema_registry()
    matches = [
        category
        for category, definition in registry.items()
        if entity_id.startswith(definition["id_prefix"])
    ]
    if not matches:
        return None
    # Prefer the longest matching prefix in case of ambiguity.
    return max(matches, key=lambda c: len(registry[c]["id_prefix"]))
