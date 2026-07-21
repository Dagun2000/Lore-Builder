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


_STATUS_EFFECT_TYPES = ("individual", "relational")


def add_status_effect(effect_id: str, label: str, type_: str = "individual", notes: str | None = None) -> None:
    """Append a new reversible status/relation to status_effects.yaml (a
    fixed set the world's rules/inference/checks all read from — imprisoned,
    cursed, ... and, since Phase 10 patch 16, target-bearing relational
    predicates like exiled/enemy_of too, distinguished by `type`) and bust
    the lru_cache so the running process sees it immediately. GUI-editable
    (and, for relational predicates, also grown automatically the first
    time inference proposes a genuinely new one — see
    pipeline_session._resolve_relational_predicates_gen), since a setting's
    cast of statuses/relations isn't something the code should hardcode — a
    future/sci-fi world might add "cryosleep" the same way a fantasy one
    added "imprisoned".

    `notes` (Phase 10 patch 22 follow-up 3) is a free-text description of
    what this status/relation actually *means* in practice — e.g. imprisoned:
    "물리적으로 수감 장소를 벗어난 행동은 불가능하다." Every LLM-facing
    context that only ever showed the bare id/label (check_status_consistency,
    the entity-context lines behind check_rule_and_notes, Step 3's own
    predicate picklist) gets this appended when present, so a rule-violation
    judgment has something concrete to reason against instead of guessing
    real-world implications from a short label alone. Optional and additive
    — an entry with no notes behaves exactly as before this field existed."""
    effect_id = (effect_id or "").strip()
    label = (label or "").strip()
    if not effect_id or not label:
        raise ValueError("id와 label은 비워둘 수 없습니다.")
    if type_ not in _STATUS_EFFECT_TYPES:
        raise ValueError(f"알 수 없는 유형입니다: {type_!r}")
    effects = load_status_effects()
    if any(e["id"] == effect_id for e in effects):
        raise ValueError(f"이미 존재하는 상태 효과 id입니다: {effect_id}")
    new_effect = {"id": effect_id, "label": label, "type": type_}
    notes = (notes or "").strip()
    if notes:
        new_effect["notes"] = notes
    _save_status_effects(effects + [new_effect])


def update_status_effect_notes(effect_id: str, notes: str | None) -> None:
    """Edit an existing status/relation's notes in place (the only field
    the GUI lets you revise post-creation — id/label/type stay fixed since
    changing them would silently orphan every timeline record already using
    this predicate). Passing an empty/None value removes the notes key
    entirely rather than storing an empty string, keeping the yaml clean."""
    effects = load_status_effects()
    notes = (notes or "").strip()
    updated = []
    found = False
    for e in effects:
        if e["id"] == effect_id:
            found = True
            e = dict(e)
            if notes:
                e["notes"] = notes
            else:
                e.pop("notes", None)
        updated.append(e)
    if not found:
        raise ValueError(f"존재하지 않는 상태 효과 id입니다: {effect_id}")
    _save_status_effects(updated)


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
    """Reverse-match an entity id (e.g. char_데이비드) to its category via id_prefix."""
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
