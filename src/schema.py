"""Schema registry loader for Lore Builder.

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


def _save_status_effects(effects: list) -> None:
    with open(STATUS_EFFECTS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"status_effects": effects}, f, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
    load_status_effects.cache_clear()


def add_status_effect(effect_id: str, label: str) -> None:
    """Append a new reversible status to status_effects.yaml (a fixed set
    the world's rules/inference/checks all read from — imprisoned, cursed,
    ...) and bust the lru_cache so the running process sees it immediately.
    A GUI-editable set, since a setting's cast of reversible statuses isn't
    something the code should hardcode — a future/sci-fi world might add
    "cryosleep" the same way a fantasy one added "imprisoned"."""
    effect_id = (effect_id or "").strip()
    label = (label or "").strip()
    if not effect_id or not label:
        raise ValueError("id와 label은 비워둘 수 없습니다.")
    effects = load_status_effects()
    if any(e["id"] == effect_id for e in effects):
        raise ValueError(f"이미 존재하는 상태 효과 id입니다: {effect_id}")
    _save_status_effects(effects + [{"id": effect_id, "label": label}])


def remove_status_effect(effect_id: str) -> None:
    """Remove a status from status_effects.yaml. Any existing timeline
    record whose predicate is this id is left untouched in storage — it
    simply stops being offered as a choice or checked for consistency going
    forward (the caller is responsible for warning about in-use ids before
    calling this, e.g. app.py's dictionary panel)."""
    effects = [e for e in load_status_effects() if e["id"] != effect_id]
    _save_status_effects(effects)


def list_categories() -> list:
    """Return every category name in schema_registry.yaml, file order."""
    return list(load_schema_registry().keys())


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


def coerce_value(field_def: dict, raw_value: str):
    """Parse a raw CLI string into the Python type a field's schema `type`
    expects. Shared by mapping.py's new-entity field review and
    detail_panel.py's existing-entity field editor."""
    if not raw_value:
        return None
    field_type = field_def["type"]
    if field_type == "integer":
        return int(raw_value)
    if field_type == "boolean":
        return raw_value.strip().lower() in ("true", "1", "예", "y", "yes")
    if field_type == "list":
        return [v.strip() for v in raw_value.split(",") if v.strip()]
    return raw_value


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
