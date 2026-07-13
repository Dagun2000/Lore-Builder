import pytest

from src import mapping, parser, storage


def test_parse_input_extracts_year_and_tags():
    text = "대륙력 2100년, [쟝]이 [검은 산양 여관]에서 얻어맞았다."

    result = parser.parse_input(text)

    assert result.year == 2100
    assert result.tags == ["쟝", "검은 산양 여관"]
    assert result.raw_text == text


def test_parse_input_raises_without_year():
    with pytest.raises(ValueError):
        parser.parse_input("[쟝]이 [검은 산양 여관]에서 얻어맞았다.")


def test_find_existing_matches_matches_seed_char_jang():
    matches = mapping.find_existing_matches("쟝", "character")

    assert matches == ["char_jang"]


def test_find_existing_matches_returns_empty_for_unknown_tag():
    matches = mapping.find_existing_matches("리나", "character")

    assert matches == []


def test_infer_terminal_status_detects_death_context(monkeypatch):
    monkeypatch.setattr(mapping, "_invoke_llm", lambda prompt: "yes")

    assert mapping.infer_terminal_status("칼에 찔려 죽었다") is True


def test_resolve_entity_creates_character_without_death_year_when_user_declines(
    monkeypatch,
):
    def fake_llm(prompt):
        return "character" if "카테고리" in prompt else "yes"

    monkeypatch.setattr(mapping, "_invoke_llm", fake_llm)
    monkeypatch.setattr(mapping, "_prompt", lambda message: "아니오")

    entity_id = mapping.resolve_entity("리나", "리나가 칼에 찔려 죽었다.", 2200)

    assert entity_id.startswith("char_")
    entity = storage.get_entity("character", entity_id)
    assert entity is not None
    assert entity["death_year"] is None


def test_create_new_entity_forces_required_field_before_saving(monkeypatch):
    # faction.category is required: true. First response is an empty Enter,
    # which must be rejected (not accepted as "skip") before a real value is
    # taken.
    responses = iter(["", "kingdom", ""])
    monkeypatch.setattr(mapping, "_prompt", lambda message: next(responses))

    entity_id = mapping._create_new_entity(
        "faction", "철혈단", "철혈단이라는 새 조직이 등장했다.", 2100
    )

    entity = storage.get_entity("faction", entity_id)
    assert entity is not None
    assert entity["category"] == "kingdom"


def test_collect_fields_rejects_clearing_a_required_field(monkeypatch):
    field_defs = mapping.schema.get_fields("faction")
    category_index = next(
        i for i, f in enumerate(field_defs, start=1) if f["name"] == "category"
    )

    # Select the required field, try to clear it (empty), then select it again
    # and provide a real value, then Enter to finish.
    responses = iter(
        [str(category_index), "", str(category_index), "mercenary_guild", ""]
    )
    monkeypatch.setattr(mapping, "_prompt", lambda message: next(responses))

    fields = mapping._collect_fields(
        "faction", preset={"category": "kingdom"}, allow_optional_review=True
    )

    assert fields["category"] == "mercenary_guild"
