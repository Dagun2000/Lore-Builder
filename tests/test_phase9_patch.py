"""Phase 9 통합 패치 — sections C (상태 연도 범위) and D (entity_presence).

C's storage/archivist pieces and the "no LLM call" gating check need no API
key (real_llm not involved, or explicitly mocked out). D's entity_presence
*classification* is a genuine LLM judgment call, so those two tests use the
real reasoning-tier model like Phase 3/5/8's e2e-style suites; the gating
*logic* itself (does main.py actually skip extra_years for entity_presence
False) is tested separately with inference.infer_relationship_and_event
mocked out, so that part stays deterministic.

Sections A, B, E (the GUI/Streamlit-facing parts of this patch) have no
automated coverage here — see the conversation for the manual verification
notes on those; pipeline_session.py's new entity_category_and_name decision
type IS covered, in test_phase8.py (it drives the same generator code the
GUI calls into).
"""

from src import archivist, inference, main, rag_check, storage
from src.inference import InferredEvent


def _script(*pairs, max_calls=15):
    calls = []

    def responder(message):
        calls.append(message)
        if len(calls) > max_calls:
            raise RuntimeError(
                f"Too many _prompt calls (possible unscripted/looping prompt). "
                f"Last message: {message!r}"
            )
        for substring, response in pairs:
            if substring in message:
                return response
        return ""

    responder.calls = calls
    return responder


# ---------------------------------------------------------------------------
# C. active_status_effects year ranges
# ---------------------------------------------------------------------------

def test_get_active_statuses_at_returns_empty_outside_range():
    storage.save_entity(
        "character",
        "char_p9c_a",
        {
            "name": "레이븐",
            "active_status_effects": [{"status": "imprisoned", "start_year": 2100, "end_year": None}],
        },
    )
    assert storage.get_active_statuses_at("char_p9c_a", 2050) == []


def test_get_active_statuses_at_returns_status_inside_range():
    storage.save_entity(
        "character",
        "char_p9c_b",
        {
            "name": "이든",
            "active_status_effects": [{"status": "imprisoned", "start_year": 2100, "end_year": None}],
        },
    )
    assert storage.get_active_statuses_at("char_p9c_b", 2150) == ["imprisoned"]


def test_get_active_statuses_at_respects_closed_end_year():
    storage.save_entity(
        "character",
        "char_p9c_c",
        {
            "name": "밀라",
            "active_status_effects": [{"status": "cursed", "start_year": 2000, "end_year": 2050}],
        },
    )
    assert storage.get_active_statuses_at("char_p9c_c", 2030) == ["cursed"]
    assert storage.get_active_statuses_at("char_p9c_c", 2080) == []


def test_check_status_consistency_skips_llm_outside_status_window(monkeypatch):
    storage.save_entity(
        "character",
        "char_p9c_d",
        {
            "name": "가론2",
            "active_status_effects": [{"status": "imprisoned", "start_year": 2100, "end_year": None}],
        },
    )
    called = []
    monkeypatch.setattr(rag_check, "_invoke_llm", lambda prompt: called.append(prompt) or "{}")

    judgment = rag_check.check_status_consistency("char_p9c_d", "가론2가 시장을 걸었다.", 2050)

    assert judgment is None
    assert called == []  # LLM never invoked — gated out before the call


def test_next_active_status_effects_set_opens_new_range():
    storage.save_entity("character", "char_p9c_e", {"name": "노바"})

    updated = archivist._next_active_status_effects("char_p9c_e", "character", "sealed", "set", 2200)

    assert updated == [{"status": "sealed", "start_year": 2200, "end_year": None}]


def test_next_active_status_effects_clear_fills_end_year():
    storage.save_entity(
        "character",
        "char_p9c_f",
        {
            "name": "아라",
            "active_status_effects": [{"status": "cursed", "start_year": 2100, "end_year": None}],
        },
    )

    updated = archivist._next_active_status_effects("char_p9c_f", "character", "cursed", "clear", 2180)

    assert updated == [{"status": "cursed", "start_year": 2100, "end_year": 2180}]


def test_seed_data_char_jang_migrated_to_range_structure():
    entity = storage.get_entity("character", "char_jang")
    ranges = entity["active_status_effects"]

    assert isinstance(ranges, list)
    imprisoned_range = next(r for r in ranges if r["status"] == "imprisoned")
    assert imprisoned_range["start_year"] == 2085
    assert "end_year" in imprisoned_range  # may be None or since-closed by another test


# ---------------------------------------------------------------------------
# D. entity_presence — "does this event imply the entity was here", not
# "is it grammatically the subject or object"
# ---------------------------------------------------------------------------

def test_infer_relationship_and_event_marks_grave_reference_not_present():
    resolved = {"밥": "char_p9d_bob", "쟝": "char_p9d_jang"}

    event = inference.infer_relationship_and_event(
        resolved, "밥이 쟝의 무덤을 파헤쳤다.", 2150
    )

    assert event.entity_presence.get("char_p9d_bob") is True
    assert event.entity_presence.get("char_p9d_jang") is False


def test_infer_relationship_and_event_marks_both_present_when_interacting():
    resolved = {"데이비드": "char_p9d_david", "쟝": "char_p9d_jang"}

    event = inference.infer_relationship_and_event(
        resolved, "데이비드가 쟝과 놀았다.", 2150
    )

    assert event.entity_presence.get("char_p9d_david") is True
    assert event.entity_presence.get("char_p9d_jang") is True


def test_hard_check_gating_respects_entity_presence(monkeypatch):
    # Deterministic version of the same scenario, with entity_presence
    # injected directly (inference mocked out) so this test doesn't depend
    # on the LLM's classification — that's covered by the two tests above.
    # Uses the terminal check (not lifespan) specifically because terminal
    # violation is purely a function of get_event_years()+extra_years vs.
    # birth/death_year — lifespan falls back to death_year directly when
    # it's set, which would make the check independent of extra_years and
    # not actually exercise the gating this test is for.
    storage.save_entity("character", "char_p9d_actor", {"name": "밥3", "birth_year": 2200})
    storage.save_entity(
        "character", "char_p9d_grave", {"name": "쟝3", "birth_year": 2000, "death_year": 2100}
    )

    fake_event = InferredEvent(
        event_summary="밥3가 쟝3의 무덤을 파헤쳤다.",
        relationships=[
            {"subject": "char_p9d_actor", "predicate": "무덤을_팠다", "object": "char_p9d_grave"}
        ],
        status_effect=None,
        entity_presence={"char_p9d_actor": True, "char_p9d_grave": False},
    )
    monkeypatch.setattr(inference, "infer_relationship_and_event", lambda *a, **k: fake_event)
    monkeypatch.setattr(main, "_prompt", _script())  # blocking rejects with no prompts

    result = main.run_pipeline_interactive("2150년, [밥3]가 [쟝3]의 무덤을 파헤쳤다.")

    conflicts_by_entity = {}
    for c in result.get("conflicts", []):
        conflicts_by_entity.setdefault(c.entity_id, []).append(c)

    # Actor: entity_presence True regardless -> this event's year (2150)
    # still applies to 밥3's own terminal check. 밥3 was "born" in 2200, so
    # a 2150 event he acted in is a blocking contradiction — proving the
    # year was actually injected for him.
    assert any(
        c.check_type == "terminal" and c.severity == "blocking"
        for c in conflicts_by_entity.get("char_p9d_actor", [])
    )
    # Referenced-but-not-present: 쟝3 is only the grave here, not asserted
    # alive at this event -> 2150 never gets applied to him, so his own
    # (already-dead-since-2100) terminal bound is never even evaluated
    # against it.
    assert conflicts_by_entity.get("char_p9d_grave", []) == []

    assert result["status"] == "rejected"
    assert result["stage"] == "hard_check"
