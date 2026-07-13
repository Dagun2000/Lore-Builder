"""Phase 5.5 — name field cleanup, done before Phase 6 needs reliable
re-identification of existing entities. No LLM calls needed here; the
prior notes/appearance-substring matching this replaces is exercised
indirectly by every e2e scenario in test_e2e.py (regression check)."""

from src import mapping, schema, storage

_NAME_BEARING_CATEGORIES = ("character", "location", "faction", "artifact", "race")


def test_name_field_added_to_five_categories():
    registry = schema.load_schema_registry()
    for category in _NAME_BEARING_CATEGORIES:
        fields = {f["name"]: f for f in registry[category]["fields"]}
        assert "name" in fields, f"{category} is missing a name field"
        assert fields["name"]["type"] == "text"
        assert fields["name"]["required"] is True


def test_new_character_name_prompt_defaults_to_tag_but_cannot_be_cleared(
    monkeypatch,
):
    # Enter accepts the tag itself as the default name (per spec's own
    # worked example: '이름 (기본값: "쟝", Enter로 그대로 사용...)').
    monkeypatch.setattr(mapping, "_prompt", lambda message: "")
    assert mapping._prompt_name("쟝") == "쟝"

    # But name is required:true — once a value exists, clearing it back to
    # empty through the free-form field review is rejected exactly like any
    # other required field (mirrors test_collect_fields_rejects_clearing_a_
    # required_field in test_phase2.py, targeted at "name" specifically).
    field_defs = schema.get_fields("character")
    name_index = next(
        i for i, f in enumerate(field_defs, start=1) if f["name"] == "name"
    )
    responses = iter([str(name_index), "", str(name_index), "쟝2", ""])
    monkeypatch.setattr(mapping, "_prompt", lambda message: next(responses))

    fields = mapping._collect_fields(
        "character", preset={"name": "쟝"}, allow_optional_review=True
    )

    assert fields["name"] == "쟝2"


def test_find_existing_matches_uses_name_field_for_char_mira():
    # char_mira's notes text doesn't literally contain "미라" in every seed
    # revision — matching must work off the dedicated name field regardless.
    matches = mapping.find_existing_matches("미라", "character")

    assert matches == ["char_mira"]


def test_find_existing_matches_partial_name_match():
    storage.save_entity(
        "character", "char_test_fontaine", {"name": "미라 폰타인"}
    )

    # "폰타인" has no exact-name match anywhere, only this partial one.
    matches = mapping.find_existing_matches("폰타인", "character")

    assert "char_test_fontaine" in matches
