"""Phase 4 (rewritten for Phase 10's event-centric redesign) uses hand-built
(not mocked) InferredEvent instances — no LLM calls happen in archivist.py,
so plain dataclass instances are enough.
"""

from src import archivist, storage
from src.inference import InferredEvent
from src.parser import ParsedInput


def test_build_diff_point_event_adds_pointers_to_every_involved_entity():
    parsed = ParsedInput(
        years=[2100],
        tags=["쟝", "검은 산양 여관"],
        raw_text="쟝이 검은 산양 여관에서 얻어맞았다.",
    )
    resolved = {"쟝": "char_쟝", "검은 산양 여관": "loc_검은_염소_주점"}
    inferred = InferredEvent(
        event_type="point",
        event_summary="쟝이 검은 산양 여관에서 얻어맞았다.",
        involved_entities=["char_쟝", "loc_검은_염소_주점"],
    )

    diff = archivist.build_diff(parsed, resolved, inferred)

    timeline_creates = [c for c in diff if c.category == "timeline" and c.action == "create"]
    pointer_updates = {c.entity_id: c for c in diff if c.action == "update"}

    assert len(timeline_creates) == 1
    assert timeline_creates[0].fields["year"] == 2100
    assert timeline_creates[0].fields["location"] == "loc_검은_염소_주점"
    timeline_id = timeline_creates[0].entity_id

    assert "char_쟝" in pointer_updates
    assert timeline_id in pointer_updates["char_쟝"].fields["event_ids"]
    assert "loc_검은_염소_주점" in pointer_updates
    assert timeline_id in pointer_updates["loc_검은_염소_주점"].fields["event_ids"]


def test_build_diff_duration_clear_updates_the_open_record_not_a_new_one():
    storage.save_entity("character", "char_p4_prisoner", {"name": "감금자"})
    storage.save_entity(
        "timeline",
        "event_p4_open_status",
        {
            "entity": "char_p4_prisoner",
            "predicate": "imprisoned",
            "target": None,
            "start_year": 2050,
            "end_year": None,
            "notes": "감금자가 2050년부터 수감 상태다.",
        },
    )
    storage.add_event_pointer("char_p4_prisoner", "event_p4_open_status")

    parsed = ParsedInput(years=[2100], tags=["감금자"], raw_text="감금자가 탈출했다.")
    resolved = {"감금자": "char_p4_prisoner"}
    inferred = InferredEvent(
        event_type="duration",
        duration_effect={
            "entity": "char_p4_prisoner",
            "predicate": "imprisoned",
            "target": None,
            "action": "clear",
            "start_year": None,
            "end_year": None,
        },
    )

    diff = archivist.build_diff(parsed, resolved, inferred)

    assert len(diff) == 1
    assert diff[0].action == "update"
    assert diff[0].category == "timeline"
    assert diff[0].entity_id == "event_p4_open_status"
    assert diff[0].fields == {"end_year": 2100}


def test_build_diff_duration_clear_with_no_open_record_returns_confirmation_needed():
    storage.save_entity("character", "char_p4_never_imprisoned", {"name": "자유인"})

    parsed = ParsedInput(years=[2100], tags=["자유인"], raw_text="자유인이 풀려났다.")
    resolved = {"자유인": "char_p4_never_imprisoned"}
    inferred = InferredEvent(
        event_type="duration",
        duration_effect={
            "entity": "char_p4_never_imprisoned",
            "predicate": "imprisoned",
            "target": None,
            "action": "clear",
            "start_year": None,
            "end_year": None,
        },
    )

    result = archivist.build_diff(parsed, resolved, inferred)

    assert isinstance(result, archivist.ConfirmationNeeded)
    assert result.reason


def test_generate_id_avoids_collision():
    existing = set()

    first = archivist.generate_id("timeline", "쟝이 아주 새로운 사건에 휘말렸다", existing)
    existing.add(first)
    second = archivist.generate_id("timeline", "쟝이 아주 새로운 사건에 휘말렸다", existing)

    assert first != second
    assert second == f"{first}_2"


def test_duration_set_always_creates_new_timeline_record():
    parsed = ParsedInput(years=[2100], tags=["쟝"], raw_text="쟝이 은빛도시와 적대하게 됐다.")
    resolved = {"쟝": "char_쟝"}
    inferred = InferredEvent(
        event_type="duration",
        duration_effect={
            "entity": "char_쟝",
            "predicate": "hostile_with",
            "target": "loc_은빛도시",
            "action": "set",
            "start_year": 2100,
            "end_year": None,
        },
    )

    diff = archivist.build_diff(parsed, resolved, inferred)

    timeline_items = [c for c in diff if c.category == "timeline"]
    assert timeline_items
    assert all(c.action == "create" for c in timeline_items)


def test_is_single_event_false_returns_confirmation_needed():
    parsed = ParsedInput(
        years=[2080, 2200], tags=["쟝", "랄프"], raw_text="쟝이 2080년에 술을 마셨고, 랄프가 2200년에 죽었다."
    )
    inferred = InferredEvent(
        event_type="point",
        is_single_event=False,
        ambiguity_reason="서로 다른 두 사건이 한 문장에 섞여 있습니다.",
    )

    result = archivist.build_diff(parsed, {"쟝": "char_쟝", "랄프": "char_ralph"}, inferred)

    assert isinstance(result, archivist.ConfirmationNeeded)
    assert result.reason == "서로 다른 두 사건이 한 문장에 섞여 있습니다."


def test_applied_duration_clear_is_immediately_visible_via_get_current_state():
    # Regression guard: get_current_state must read the same event_ids the
    # archivist's update ChangeItem targets, not a separate snapshot field —
    # otherwise a saved diff can look like it silently didn't take effect.
    storage.save_entity("character", "char_p4_prisoner2", {"name": "감금자2"})
    storage.save_entity(
        "timeline",
        "event_p4_open_status2",
        {
            "entity": "char_p4_prisoner2",
            "predicate": "imprisoned",
            "target": None,
            "start_year": 2050,
            "end_year": None,
            "notes": "감금자2가 2050년부터 수감 상태다.",
        },
    )
    storage.add_event_pointer("char_p4_prisoner2", "event_p4_open_status2")

    assert storage.get_current_state("char_p4_prisoner2", "imprisoned")

    parsed = ParsedInput(years=[2100], tags=["감금자2"], raw_text="감금자2가 탈출했다.")
    resolved = {"감금자2": "char_p4_prisoner2"}
    inferred = InferredEvent(
        event_type="duration",
        duration_effect={
            "entity": "char_p4_prisoner2",
            "predicate": "imprisoned",
            "target": None,
            "action": "clear",
            "start_year": None,
            "end_year": None,
        },
    )

    diff = archivist.build_diff(parsed, resolved, inferred)
    update_item = next(c for c in diff if c.category == "timeline" and c.action == "update")

    # Apply the diff item the way Phase 5's approval loop will.
    storage.save_entity(update_item.category, update_item.entity_id, update_item.fields)

    assert not storage.get_current_state("char_p4_prisoner2", "imprisoned")
