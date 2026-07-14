"""Phase 6.5 patch — dedupe the flag list + auto-clear on fix.

Pure bookkeeping (no LLM), so no API key needed, same as test_phase6_5.py.
"""

from src import field_update, flags, storage


def test_list_flags_deduped_collapses_repeat_entity_id():
    flags.add_flag("char_p65p_a", "맥락1")
    flags.add_flag("char_p65p_a", "맥락2")
    flags.add_flag("char_p65p_a", "맥락3")

    matching = [f for f in flags.list_flags_deduped() if f.entity_id == "char_p65p_a"]
    assert len(matching) == 1


def test_clear_flags_for_entity_removes_all_regardless_of_context():
    flags.add_flag("char_p65p_b", "맥락1")
    flags.add_flag("char_p65p_b", "맥락2")
    flags.add_flag("char_p65p_b", "맥락3")

    cleared = flags.clear_flags_for_entity("char_p65p_b")

    assert cleared == 3
    assert not any(f.entity_id == "char_p65p_b" for f in flags.list_flags())


def test_update_field_flow_auto_clears_flags_on_the_edited_entity(monkeypatch):
    storage.save_entity("character", "char_p65p_c", {"name": "다림"})
    flags.add_flag("char_p65p_c", "맥락1")
    flags.add_flag("char_p65p_c", "맥락2")
    assert any(f.entity_id == "char_p65p_c" for f in flags.list_flags_deduped())

    monkeypatch.setattr(field_update, "_prompt", lambda message: "y")

    result = field_update.update_field_flow("char_p65p_c", "appearance", "새로운 외모.")

    assert result["status"] == "saved"
    assert result["cleared_flags"] == 2
    assert not any(f.entity_id == "char_p65p_c" for f in flags.list_flags_deduped())
    assert not any(f.entity_id == "char_p65p_c" for f in flags.list_flags())


def test_list_flags_and_clear_flag_still_work_unmodified():
    flag1 = flags.add_flag("char_p65p_d", "맥락1", "사유1")
    flag2 = flags.add_flag("char_p65p_d", "맥락2", "사유2")

    all_flags = flags.list_flags()
    assert any(f.id == flag1.id and f.reason == "사유1" for f in all_flags)
    assert any(f.id == flag2.id and f.reason == "사유2" for f in all_flags)
    # list_flags() itself must NOT be deduped — both occurrences still show.
    assert len([f for f in all_flags if f.entity_id == "char_p65p_d"]) == 2

    flags.clear_flag(flag1.id)

    remaining = [f for f in flags.list_flags() if f.entity_id == "char_p65p_d"]
    assert len(remaining) == 1
    assert remaining[0].id == flag2.id
