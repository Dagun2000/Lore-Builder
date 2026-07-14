"""Phase 10 — event-centric redesign (relationship retired, event_ids +
point/duration timeline records, deletion).

Sections 3 (archivist.build_diff) and 5 (field_update.find_related_context)
already have full coverage in the rewritten test_phase4.py/test_phase6.py —
not duplicated here. This file covers section 1 (parser), section 2
(inference judgment, real LLM), section 4 (storage query layer), section 6
(deletion), and the Phase 8 integration point for ConfirmationNeeded
(the new "multi_event_warning" pending_decision).
"""

from src import archivist, deletion, inference, parser, pipeline_session, storage
from src.inference import InferredEvent


# ---------------------------------------------------------------------------
# 1. parser.py — multi-year extraction, bracket exclusion
# ---------------------------------------------------------------------------

def test_parse_input_extracts_multiple_years():
    result = parser.parse_input("2000년부터 2010년까지 봉인되었다.")
    assert result.years == [2000, 2010]


def test_parse_input_ignores_years_inside_brackets():
    import pytest

    with pytest.raises(ValueError):
        parser.parse_input("[100년 전쟁]에서 사건이 있었다.")


def test_parse_input_extracts_only_the_year_outside_brackets():
    result = parser.parse_input("2010년, [100년 전쟁] 중 사건이 있었다.")
    assert result.years == [2010]


# ---------------------------------------------------------------------------
# 2. inference.py — point vs duration (set/clear/set_closed) judgment,
# real LLM per project convention.
# ---------------------------------------------------------------------------

def test_infer_event_duration_set():
    result = inference.infer_event({"쟝": "char_jang"}, "쟝이 2000년에 봉인되었다.", [2000])

    assert result.is_single_event
    assert result.event_type == "duration"
    assert result.duration_effect["action"] == "set"
    assert result.duration_effect["predicate"] == "sealed"
    assert result.duration_effect["start_year"] == 2000
    assert result.duration_effect["end_year"] is None


def test_infer_event_duration_clear():
    result = inference.infer_event({"쟝": "char_jang"}, "쟝이 2010년에 봉인에서 풀려났다.", [2010])

    assert result.is_single_event
    assert result.event_type == "duration"
    assert result.duration_effect["action"] == "clear"
    assert result.duration_effect["predicate"] == "sealed"


def test_infer_event_duration_set_closed():
    result = inference.infer_event(
        {"쟝": "char_jang"}, "쟝은 2000년부터 2010년까지 봉인되었다.", [2000, 2010]
    )

    assert result.is_single_event
    assert result.event_type == "duration"
    assert result.duration_effect["action"] == "set_closed"
    assert result.duration_effect["start_year"] == 2000
    assert result.duration_effect["end_year"] == 2010


def test_infer_event_point_for_mundane_action():
    resolved = {"쟝": "char_jang", "검은 염소 주점": "loc_black_goat_inn"}
    result = inference.infer_event(resolved, "쟝이 주점에서 술을 마셨다.", [2100])

    assert result.is_single_event
    assert result.event_type == "point"


def test_infer_event_marks_ambiguous_multi_subject_sentence():
    resolved = {"쟝": "char_jang", "늙은 왕": "char_old_king"}
    result = inference.infer_event(
        resolved, "쟝이 2080년에 술을 마셨고, 늙은 왕이 1550년에 죽었다.", [1550, 2080]
    )

    assert result.is_single_event is False
    assert result.ambiguity_reason


# ---------------------------------------------------------------------------
# Phase 8 integration — ConfirmationNeeded surfaces as multi_event_warning
# ---------------------------------------------------------------------------

def test_ambiguous_input_pauses_as_multi_event_warning(monkeypatch):
    fake_event = InferredEvent(
        event_type="point",
        is_single_event=False,
        ambiguity_reason="테스트용 모호함 사유.",
    )
    monkeypatch.setattr(inference, "infer_event", lambda *a, **k: fake_event)

    session = pipeline_session.start_session("2100년, [쟝]이 무언가를 했다.")

    assert session.pending_decision.decision_type == "multi_event_warning"
    assert session.pending_decision.payload["reason"] == "테스트용 모호함 사유."

    session = pipeline_session.resume_session(session.session_id, False)

    assert session.pending_decision is None
    assert session.stage == "aborted"
    assert session.result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# 4. storage.py — event_ids-based query layer
# ---------------------------------------------------------------------------

def test_get_duration_records_matches_entity_and_target_sides():
    storage.save_entity("character", "char_p10_a", {"name": "무명A"})
    storage.save_entity("character", "char_p10_b", {"name": "무명B"})
    storage.save_entity(
        "timeline",
        "event_p10_knows",
        {
            "entity": "char_p10_a",
            "predicate": "knows",
            "target": "char_p10_b",
            "start_year": 2000,
            "end_year": None,
        },
    )
    storage.add_event_pointer("char_p10_a", "event_p10_knows")
    storage.add_event_pointer("char_p10_b", "event_p10_knows")

    assert [r["id"] for r in storage.get_duration_records("char_p10_a")] == ["event_p10_knows"]
    assert [r["id"] for r in storage.get_duration_records("char_p10_b")] == ["event_p10_knows"]
    assert [r["id"] for r in storage.get_duration_records("char_p10_a", "knows")] == ["event_p10_knows"]
    assert storage.get_duration_records("char_p10_a", "hostile_with") == []


def test_get_current_state_bounded_by_year():
    storage.save_entity("character", "char_p10_c", {"name": "무명C"})
    storage.save_entity(
        "timeline",
        "event_p10_status",
        {
            "entity": "char_p10_c",
            "predicate": "cursed",
            "target": None,
            "start_year": 2000,
            "end_year": 2050,
        },
    )
    storage.add_event_pointer("char_p10_c", "event_p10_status")

    assert storage.get_current_state("char_p10_c", "cursed", 2030)
    assert not storage.get_current_state("char_p10_c", "cursed", 2080)
    assert not storage.get_current_state("char_p10_c", "cursed", 1990)


def test_get_events_for_entity_sorts_point_and_duration_chronologically():
    storage.save_entity("character", "char_p10_d", {"name": "무명D"})
    storage.save_entity("timeline", "event_p10_d_point", {"year": 2100, "notes": "점 이벤트"})
    storage.save_entity(
        "timeline",
        "event_p10_d_duration",
        {"entity": "char_p10_d", "predicate": "knows", "target": None, "start_year": 2050, "end_year": None},
    )
    storage.add_event_pointer("char_p10_d", "event_p10_d_point")
    storage.add_event_pointer("char_p10_d", "event_p10_d_duration")

    records = storage.get_events_for_entity("char_p10_d")

    assert [r["id"] for r in records] == ["event_p10_d_duration", "event_p10_d_point"]


# ---------------------------------------------------------------------------
# 6. deletion.py
# ---------------------------------------------------------------------------

def test_delete_event_removes_pointer_from_every_participant():
    storage.save_entity("character", "char_p10_e", {"name": "케인"})
    storage.save_entity("character", "char_p10_f", {"name": "애슐리"})
    storage.save_entity("location", "loc_p10_e", {"name": "여관", "category": "tavern"})
    storage.save_entity("timeline", "event_p10_e_shared", {"year": 2100, "notes": "셋이 함께한 사건."})
    for entity_id in ("char_p10_e", "char_p10_f", "loc_p10_e"):
        storage.add_event_pointer(entity_id, "event_p10_e_shared")

    result = deletion.delete_event("event_p10_e_shared")

    assert set(result.affected_entities) == {"char_p10_e", "char_p10_f", "loc_p10_e"}
    assert storage.get_entity("timeline", "event_p10_e_shared") is None
    for entity_id, category in (("char_p10_e", "character"), ("char_p10_f", "character"), ("loc_p10_e", "location")):
        assert "event_p10_e_shared" not in (storage.get_entity(category, entity_id).get("event_ids") or [])


def test_request_entity_deletion_returns_full_event_content():
    storage.save_entity("character", "char_p10_g", {"name": "무명G"})
    storage.save_entity("timeline", "event_p10_g_a", {"year": 2000, "notes": "사건 A"})
    storage.save_entity("timeline", "event_p10_g_b", {"year": 2010, "notes": "사건 B"})
    storage.add_event_pointer("char_p10_g", "event_p10_g_a")
    storage.add_event_pointer("char_p10_g", "event_p10_g_b")

    events = deletion.request_entity_deletion("char_p10_g")

    assert {e["id"] for e in events} == {"event_p10_g_a", "event_p10_g_b"}
    assert {e["notes"] for e in events} == {"사건 A", "사건 B"}


def test_delete_entity_cascades_orphaned_event_but_keeps_shared_event():
    storage.save_entity("character", "char_p10_h", {"name": "무명H"})
    storage.save_entity("character", "char_p10_i", {"name": "무명I"})

    # Sole participant -> should cascade-delete when char_p10_h is deleted.
    storage.save_entity(
        "timeline",
        "event_p10_h_solo_status",
        {"entity": "char_p10_h", "predicate": "cursed", "target": None, "start_year": 2000, "end_year": None},
    )
    storage.add_event_pointer("char_p10_h", "event_p10_h_solo_status")

    # Shared with char_p10_i -> should survive, just drop char_p10_h's pointer.
    storage.save_entity("timeline", "event_p10_h_shared", {"year": 2050, "notes": "함께한 사건"})
    storage.add_event_pointer("char_p10_h", "event_p10_h_shared")
    storage.add_event_pointer("char_p10_i", "event_p10_h_shared")

    result = deletion.delete_entity("char_p10_h", "character")

    assert storage.get_entity("character", "char_p10_h") is None
    assert "event_p10_h_solo_status" in result.deleted_events
    assert storage.get_entity("timeline", "event_p10_h_solo_status") is None

    assert "event_p10_h_shared" not in result.deleted_events
    assert storage.get_entity("timeline", "event_p10_h_shared") is not None
    assert "event_p10_h_shared" in (storage.get_entity("character", "char_p10_i").get("event_ids") or [])
