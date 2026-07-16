"""Phase 6.5 — flagging related records for later review.

Flags are pure bookkeeping (no LLM, no validation) so these tests don't
need an API key.
"""

from src import field_update, flags, storage


def test_add_flag_appears_in_list_flags():
    flag = flags.add_flag("char_p65_a", "테스트 맥락", "확인 필요")

    all_flags = flags.list_flags()
    match = next(f for f in all_flags if f.id == flag.id)

    assert match.entity_id == "char_p65_a"
    assert match.flagged_from == "테스트 맥락"
    assert match.reason == "확인 필요"


def test_add_flag_allows_duplicate_entity_id():
    flag1 = flags.add_flag("char_p65_b", "맥락1")
    flag2 = flags.add_flag("char_p65_b", "맥락2")

    assert flag1.id != flag2.id
    matching = [f for f in flags.list_flags() if f.entity_id == "char_p65_b"]
    assert len(matching) == 2


def test_clear_flag_removes_it():
    flag = flags.add_flag("char_p65_c", "테스트 맥락")
    assert any(f.id == flag.id for f in flags.list_flags())

    flags.clear_flag(flag.id)

    assert not any(f.id == flag.id for f in flags.list_flags())


def test_update_field_flow_flagging_is_independent_of_field_save(monkeypatch):
    storage.save_entity("character", "char_p65_d", {"name": "가름"})
    storage.save_entity(
        "timeline", "event_p65_d", {"year": 2050, "notes": "가름과 관련된 무관한 사건."}
    )
    storage.add_event_pointer("char_p65_d", "event_p65_d")

    # 1 -> flag the first (only) related record, then a reason, then save.
    responses = iter(["1", "다시 확인 필요", "y"])
    monkeypatch.setattr(field_update, "_prompt", lambda message: next(responses))

    result = field_update.update_field_flow("char_p65_d", "notes", "새로운 외모.")

    assert result["status"] == "saved"
    assert storage.get_entity("character", "char_p65_d")["notes"] == "새로운 외모."

    assert len(result["flagged"]) == 1
    flagged_entry = result["flagged"][0]
    assert flagged_entry.entity_id == "event_p65_d"
    assert flagged_entry.reason == "다시 확인 필요"
    assert any(f.entity_id == "event_p65_d" for f in flags.list_flags())


def test_update_field_flow_skipping_flag_prompt_still_saves_normally(monkeypatch):
    storage.save_entity("character", "char_p65_e", {"name": "누리"})
    storage.save_entity(
        "timeline", "event_p65_e", {"year": 2050, "notes": "누리와 관련된 무관한 사건."}
    )
    storage.add_event_pointer("char_p65_e", "event_p65_e")

    # Enter to skip flagging entirely, then y to save — proves the flag
    # prompt being skipped doesn't disrupt the pre-existing Phase 6 flow.
    responses = iter(["", "y"])
    monkeypatch.setattr(field_update, "_prompt", lambda message: next(responses))

    result = field_update.update_field_flow("char_p65_e", "notes", "평범한 외모.")

    assert result["status"] == "saved"
    assert storage.get_entity("character", "char_p65_e")["notes"] == "평범한 외모."
    assert result["flagged"] == []
