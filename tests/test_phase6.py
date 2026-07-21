"""Phase 6 — existing-entity field update + related-record listing.

Phase 10 replaced find_related_context's relationship-table-plus-Chroma-
similarity search with a plain event_ids listing — no ranking, no LLM call,
so these tests still don't need an API key.
"""

from src import field_update, storage


def test_update_structured_field_reuses_phase1_lifespan_warning(monkeypatch):
    storage.save_entity("race", "race_p6_short", {"name": "단명종", "lifespan": 50})
    storage.save_entity("character", "char_p6_a", {"name": "가온", "birth_year": 2000})
    storage.save_entity("timeline", "event_p6_a", {"year": 2100})
    storage.add_event_pointer("char_p6_a", "event_p6_a")

    monkeypatch.setattr(field_update.approval, "_prompt", lambda message: "그래도 저장")

    approved, conflicts = field_update.update_structured_field(
        "char_p6_a", "race", "race_p6_short"
    )

    assert approved is True
    assert any(c.check_type == "lifespan" for c in conflicts)
    entity = storage.get_entity("character", "char_p6_a")
    assert entity["race"] == "race_p6_short"
    assert entity["lifespan_check_ack"]


def test_find_related_context_lists_every_pointed_at_event():
    storage.save_entity("character", "char_p6_b", {"name": "레인"})

    storage.save_entity(
        "timeline",
        "event_p6_b_membership",
        {
            "entity": "char_p6_b",
            "predicate": "member_of",
            "target": "faction_용병_길드",
            "start_year": 2040,
            "end_year": None,
            "notes": "레인이 용병 길드에 들어갔다.",
        },
    )
    storage.add_event_pointer("char_p6_b", "event_p6_b_membership")

    storage.save_entity(
        "timeline", "event_p6_b_duel", {"year": 2050, "notes": "레인과 관련된 유일한 사건 기록."}
    )
    storage.add_event_pointer("char_p6_b", "event_p6_b_duel")

    docs = field_update.find_related_context("char_p6_b")

    by_id = {d.entity_id: d for d in docs}
    assert by_id["event_p6_b_membership"].source == "duration"
    assert by_id["event_p6_b_membership"].relation == "member_of"
    assert by_id["event_p6_b_duel"].source == "point"
    assert by_id["event_p6_b_duel"].relation == "event"
    # Chronological, not similarity-ranked: membership (2040) before duel (2050).
    assert by_id["event_p6_b_membership"].relevance_rank < by_id["event_p6_b_duel"].relevance_rank


def test_find_related_context_drops_nothing_regardless_of_count():
    storage.save_entity("character", "char_p6_c", {"name": "노엘"})
    event_ids = []
    for i in range(4):
        event_id = f"event_p6_c_{i}"
        storage.save_entity("timeline", event_id, {"year": 2000 + i, "notes": f"노엘 관련 사건 {i}"})
        storage.add_event_pointer("char_p6_c", event_id)
        event_ids.append(event_id)

    docs = field_update.find_related_context("char_p6_c")

    assert {d.entity_id for d in docs} == set(event_ids)


def test_update_field_flow_race_triggers_both_tracks(monkeypatch):
    storage.save_entity("race", "race_p6_short2", {"name": "단명종2", "lifespan": 50})
    storage.save_entity("character", "char_p6_d", {"name": "다온", "birth_year": 2000})
    storage.save_entity("timeline", "event_p6_d", {"year": 2100})
    storage.add_event_pointer("char_p6_d", "event_p6_d")
    storage.save_entity(
        "timeline",
        "event_p6_d_rejection",
        {
            "entity": "faction_p6_d",
            "predicate": "rejects",
            "target": "char_p6_d",
            "start_year": 2090,
            "end_year": None,
            "notes": "단명종2는 절대 받아주지 않는다.",
        },
    )
    storage.save_entity("faction", "faction_p6_d", {"name": "단명종반대", "category": "tribe"})
    storage.add_event_pointer("char_p6_d", "event_p6_d_rejection")
    storage.add_event_pointer("faction_p6_d", "event_p6_d_rejection")

    monkeypatch.setattr(field_update, "_prompt", lambda message: "y")
    monkeypatch.setattr(field_update.approval, "_prompt", lambda message: "그래도 저장")

    result = field_update.update_field_flow("char_p6_d", "race", "race_p6_short2")

    assert result["status"] == "saved"
    assert any(c.check_type == "lifespan" for c in result["conflicts"])  # Track A ran
    assert any(d.entity_id == "event_p6_d_rejection" for d in result["related_docs"])  # Track B ran
    assert storage.get_entity("character", "char_p6_d")["race"] == "race_p6_short2"


def test_update_field_flow_with_no_related_context_still_saves(monkeypatch):
    storage.save_entity("character", "char_p6_e", {"name": "우나"})

    monkeypatch.setattr(field_update, "_prompt", lambda message: "y")

    result = field_update.update_field_flow(
        "char_p6_e", "notes", "키가 크고 마른 체형."
    )

    assert result["status"] == "saved"
    assert result["related_docs"] == []
    entity = storage.get_entity("character", "char_p6_e")
    assert entity["notes"] == "키가 크고 마른 체형."


def test_update_field_flow_more_reveals_remaining_docs(monkeypatch, capsys):
    storage.save_entity("character", "char_p6_f", {"name": "핀"})
    for i in range(5):
        event_id = f"event_p6_more_{i}"
        text = f"핀 관련 사건 {i}"
        storage.save_entity("timeline", event_id, {"year": 2000 + i, "notes": text})
        storage.add_event_pointer("char_p6_f", event_id)

    # "more" -> expand the rest, "" -> skip flagging (Phase 6.5), "y" -> save.
    responses = iter(["more", "", "y"])
    monkeypatch.setattr(field_update, "_prompt", lambda message: next(responses))

    result = field_update.update_field_flow("char_p6_f", "notes", "평범한 외모.")

    captured = capsys.readouterr()
    assert result["status"] == "saved"
    assert len(result["related_docs"]) == 5
    for i in range(5):
        assert f"event_p6_more_{i}" in captured.out
