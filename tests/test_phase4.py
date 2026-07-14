"""Phase 4 uses hand-built (not mocked) InferredEvent/Judgment instances —
no LLM calls happen in archivist.py, so plain dataclass instances are enough.
"""

from src import archivist, storage
from src.inference import InferredEvent
from src.parser import ParsedInput
from src.rag_check import Judgment


def test_build_diff_creates_timeline_and_relationship():
    parsed = ParsedInput(
        year=2100,
        tags=["쟝", "검은 산양 여관"],
        raw_text="쟝이 검은 산양 여관에서 얻어맞았다.",
    )
    resolved = {"쟝": "char_jang", "검은 산양 여관": "loc_black_goat_inn"}
    inferred = InferredEvent(
        event_summary="쟝이 검은 산양 여관에서 얻어맞았다.",
        relationships=[
            {
                "subject": "char_jang",
                "predicate": "얻어맞음_장소",
                "object": "loc_black_goat_inn",
            }
        ],
        status_effect=None,
    )

    diff = archivist.build_diff(parsed, resolved, inferred, [])

    timeline_creates = [c for c in diff if c.category == "timeline" and c.action == "create"]
    relationship_creates = [
        c for c in diff if c.category == "relationship" and c.action == "create"
    ]

    assert len(timeline_creates) == 1
    assert timeline_creates[0].fields["year"] == 2100
    assert timeline_creates[0].fields["location"] == "loc_black_goat_inn"

    assert len(relationship_creates) == 1
    assert relationship_creates[0].fields["subject"] == "char_jang"
    assert relationship_creates[0].fields["object"] == "loc_black_goat_inn"


def test_build_diff_status_clear_produces_character_update():
    parsed = ParsedInput(
        year=2100, tags=["쟝"], raw_text="쟝이 탈출해서 마을로 도망쳤다."
    )
    resolved = {"쟝": "char_jang"}
    inferred = InferredEvent(
        event_summary="쟝이 수감 상태에서 탈출했다.",
        relationships=[],
        status_effect={"entity": "char_jang", "effect": "imprisoned", "action": "clear"},
    )
    judgment = Judgment(
        type="clears_status",
        reason="쟝이 탈출했다는 문장은 수감 상태 해제를 의미함",
        entity_id="char_jang",
        status_effect_id="imprisoned",
    )

    diff = archivist.build_diff(parsed, resolved, inferred, [judgment])

    timeline_creates = [c for c in diff if c.category == "timeline" and c.action == "create"]
    character_updates = [c for c in diff if c.category == "character" and c.action == "update"]

    assert len(timeline_creates) == 1
    assert timeline_creates[0].fields["status_effect"] is None  # clearing, not setting

    assert len(character_updates) == 1
    assert character_updates[0].entity_id == "char_jang"
    open_statuses = [
        r["status"]
        for r in character_updates[0].fields["active_status_effects"]
        if r.get("end_year") is None
    ]
    assert "imprisoned" not in open_statuses


def test_generate_id_avoids_collision():
    existing = set()

    first = archivist.generate_id("timeline", "쟝이 아주 새로운 사건에 휘말렸다", existing)
    existing.add(first)
    second = archivist.generate_id("timeline", "쟝이 아주 새로운 사건에 휘말렸다", existing)

    assert first != second
    assert second == f"{first}_2"


def test_relationships_are_always_create_never_update():
    parsed = ParsedInput(
        year=2100, tags=["쟝"], raw_text="쟝이 탈출해서 마을로 도망쳤다."
    )
    resolved = {"쟝": "char_jang"}
    inferred = InferredEvent(
        event_summary="쟝이 수감 상태에서 탈출했다.",
        relationships=[
            {"subject": "char_jang", "predicate": "탈출함", "object": "char_jang"}
        ],
        status_effect={"entity": "char_jang", "effect": "imprisoned", "action": "clear"},
    )

    diff = archivist.build_diff(parsed, resolved, inferred, [])

    relationship_items = [c for c in diff if c.category == "relationship"]

    assert relationship_items
    assert all(c.action == "create" for c in relationship_items)


def test_applied_status_update_is_immediately_visible_via_get_status_effects():
    # Regression guard: get_status_effects must read the same
    # active_status_effects field archivist writes to, not a separate
    # timeline/relationship history scan — otherwise a saved diff can look
    # like it silently didn't take effect.
    assert "imprisoned" in storage.get_status_effects("char_jang")

    parsed = ParsedInput(
        year=2100, tags=["쟝"], raw_text="쟝이 탈출해서 마을로 도망쳤다."
    )
    resolved = {"쟝": "char_jang"}
    inferred = InferredEvent(
        event_summary="쟝이 수감 상태에서 탈출했다.",
        relationships=[],
        status_effect={"entity": "char_jang", "effect": "imprisoned", "action": "clear"},
    )

    diff = archivist.build_diff(parsed, resolved, inferred, [])
    character_update = next(
        c for c in diff if c.category == "character" and c.action == "update"
    )

    # Apply the diff item the way Phase 5's approval loop will.
    storage.save_entity(
        character_update.category, character_update.entity_id, character_update.fields
    )

    assert "imprisoned" not in storage.get_status_effects("char_jang")
