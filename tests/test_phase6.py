"""Phase 6 — existing-entity field update + related-record search.

find_related_context does pure retrieval + ranking (Chroma similarity over
a fixed candidate pool) — no LLM call, so these tests don't need an API key,
unlike Phase 3/5's judgment-calling tests.
"""

from src import field_update, storage


def test_update_structured_field_reuses_phase1_lifespan_warning(monkeypatch):
    storage.save_entity("race", "race_p6_short", {"name": "단명종", "lifespan": 50})
    storage.save_entity("character", "char_p6_a", {"name": "가온", "birth_year": 2000})
    storage.save_entity("timeline", "event_p6_a", {"year": 2100})
    storage.save_entity(
        "relationship",
        "rel_p6_a",
        {"subject": "char_p6_a", "predicate": "involved_in", "object": "event_p6_a"},
    )

    monkeypatch.setattr(field_update.approval, "_prompt", lambda message: "그래도 저장")

    approved, conflicts = field_update.update_structured_field(
        "char_p6_a", "race", "race_p6_short"
    )

    assert approved is True
    assert any(c.check_type == "lifespan" for c in conflicts)
    entity = storage.get_entity("character", "char_p6_a")
    assert entity["race"] == "race_p6_short"
    assert entity["lifespan_check_ack"]


def test_find_related_context_full_recall_relationship_and_timeline():
    # Reachable ONLY via the relationship bucket (counterpart is a real
    # entity, not an event).
    storage.save_entity(
        "faction",
        "faction_p6_guild",
        {"name": "P6길드", "category": "mercenary_guild", "notes": "여성만 가입 가능."},
    )
    storage.save_to_chroma("faction_p6_guild", "P6길드는 여성 전용 길드다.", {"category": "faction"})

    # Reachable ONLY via the timeline bucket (relationship-mediated to an
    # event_, per find_related_timeline_ids' indirect path).
    storage.save_entity("character", "char_p6_b", {"name": "레인"})
    storage.save_entity(
        "timeline", "event_p6_b", {"year": 2050, "notes": "레인과 관련된 유일한 사건 기록."}
    )
    storage.save_to_chroma("event_p6_b", "레인과 관련된 유일한 사건 기록.", {"category": "timeline"})

    storage.save_entity(
        "relationship",
        "rel_p6_guild",
        {"subject": "char_p6_b", "predicate": "member_of", "object": "faction_p6_guild"},
    )
    storage.save_entity(
        "relationship",
        "rel_p6_event",
        {"subject": "char_p6_b", "predicate": "involved_in", "object": "event_p6_b"},
    )

    docs = field_update.find_related_context("char_p6_b", "gender", "남성")

    by_id = {d.entity_id: d for d in docs}
    assert by_id["faction_p6_guild"].source == "relationship"
    assert by_id["event_p6_b"].source == "timeline"


def test_find_related_context_ranks_gender_synonyms_without_dropping_any():
    storage.save_entity("character", "char_p6_c", {"name": "노엘"})

    storage.save_entity(
        "faction",
        "faction_p6_women",
        {"name": "여성단", "category": "tribe", "notes": "여성 전용으로 운영되는 부족이다."},
    )
    storage.save_to_chroma(
        "faction_p6_women", "여성 전용으로 운영되는 부족이다.", {"category": "faction"}
    )
    storage.save_entity(
        "relationship",
        "rel_p6_c_women",
        {"subject": "char_p6_c", "predicate": "related_to", "object": "faction_p6_women"},
    )

    storage.save_entity(
        "timeline", "event_p6_c_duel", {"year": 2060, "notes": "노엘이 남자들만의 결투에 참가했다."}
    )
    storage.save_to_chroma(
        "event_p6_c_duel", "노엘이 남자들만의 결투에 참가했다.", {"category": "timeline"}
    )
    storage.save_entity(
        "relationship",
        "rel_p6_c_duel",
        {"subject": "char_p6_c", "predicate": "involved_in", "object": "event_p6_c_duel"},
    )

    storage.save_entity(
        "faction",
        "faction_p6_lady",
        {"name": "귀부인회", "category": "religious_order", "notes": "이 조직은 여인들만 받아들인다."},
    )
    storage.save_to_chroma(
        "faction_p6_lady", "이 조직은 여인들만 받아들인다.", {"category": "faction"}
    )
    storage.save_entity(
        "relationship",
        "rel_p6_c_lady",
        {"subject": "char_p6_c", "predicate": "related_to", "object": "faction_p6_lady"},
    )

    storage.save_entity(
        "location",
        "loc_p6_unrelated",
        {"name": "평범한 시장", "category": "city", "notes": "특별한 성별 조건이 없는 평범한 시장이다."},
    )
    storage.save_to_chroma(
        "loc_p6_unrelated", "특별한 성별 조건이 없는 평범한 시장이다.", {"category": "location"}
    )
    storage.save_entity(
        "relationship",
        "rel_p6_c_market",
        {"subject": "char_p6_c", "predicate": "visited", "object": "loc_p6_unrelated"},
    )

    docs = field_update.find_related_context("char_p6_c", "성별", "남성")

    doc_ids = {d.entity_id for d in docs}
    for expected_id in (
        "faction_p6_women",
        "event_p6_c_duel",
        "faction_p6_lady",
        "loc_p6_unrelated",
    ):
        assert expected_id in doc_ids, f"{expected_id} missing from ranked results"


def test_update_field_flow_race_triggers_both_tracks(monkeypatch):
    storage.save_entity("race", "race_p6_short2", {"name": "단명종2", "lifespan": 50})
    storage.save_entity("character", "char_p6_d", {"name": "다온", "birth_year": 2000})
    storage.save_entity("timeline", "event_p6_d", {"year": 2100})
    storage.save_entity(
        "relationship",
        "rel_p6_d",
        {"subject": "char_p6_d", "predicate": "involved_in", "object": "event_p6_d"},
    )
    storage.save_entity(
        "faction",
        "faction_p6_d",
        {"name": "단명종반대", "category": "tribe", "notes": "단명종2는 절대 받아주지 않는다."},
    )
    storage.save_to_chroma(
        "faction_p6_d", "단명종2는 절대 받아주지 않는다.", {"category": "faction"}
    )
    storage.save_entity(
        "relationship",
        "rel_p6_d_faction",
        {"subject": "char_p6_d", "predicate": "related_to", "object": "faction_p6_d"},
    )

    monkeypatch.setattr(field_update, "_prompt", lambda message: "y")
    monkeypatch.setattr(field_update.approval, "_prompt", lambda message: "그래도 저장")

    result = field_update.update_field_flow("char_p6_d", "race", "race_p6_short2")

    assert result["status"] == "saved"
    assert any(c.check_type == "lifespan" for c in result["conflicts"])  # Track A ran
    assert any(d.entity_id == "faction_p6_d" for d in result["related_docs"])  # Track B ran
    assert storage.get_entity("character", "char_p6_d")["race"] == "race_p6_short2"


def test_update_field_flow_with_no_related_context_still_saves(monkeypatch):
    storage.save_entity("character", "char_p6_e", {"name": "우나"})

    monkeypatch.setattr(field_update, "_prompt", lambda message: "y")

    result = field_update.update_field_flow(
        "char_p6_e", "appearance", "키가 크고 마른 체형."
    )

    assert result["status"] == "saved"
    assert result["related_docs"] == []
    entity = storage.get_entity("character", "char_p6_e")
    assert entity["appearance"] == "키가 크고 마른 체형."


def test_update_field_flow_more_reveals_remaining_docs(monkeypatch, capsys):
    storage.save_entity("character", "char_p6_f", {"name": "핀"})
    for i in range(5):
        event_id = f"event_p6_more_{i}"
        text = f"핀 관련 사건 {i}"
        storage.save_entity("timeline", event_id, {"year": 2000 + i, "notes": text})
        storage.save_to_chroma(event_id, text, {"category": "timeline"})
        storage.save_entity(
            "relationship",
            f"rel_p6_more_{i}",
            {"subject": "char_p6_f", "predicate": "involved_in", "object": event_id},
        )

    # "more" -> expand the rest, "" -> skip flagging (Phase 6.5), "y" -> save.
    responses = iter(["more", "", "y"])
    monkeypatch.setattr(field_update, "_prompt", lambda message: next(responses))

    result = field_update.update_field_flow("char_p6_f", "appearance", "평범한 외모.")

    captured = capsys.readouterr()
    assert result["status"] == "saved"
    assert len(result["related_docs"]) == 5
    for i in range(5):
        assert f"event_p6_more_{i}" in captured.out
