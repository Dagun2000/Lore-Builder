from src import hard_check, storage


def _link_event(entity_id: str, event_id: str, year: int) -> None:
    storage.save_entity("timeline", event_id, {"year": year})
    storage.add_event_pointer(entity_id, event_id)


def test_terminal_check_passes_when_no_death_year():
    storage.save_entity("character", "char_t1_alive", {"birth_year": 2000})
    _link_event("char_t1_alive", "event_t1_a", 2050)

    assert hard_check.check_terminal_violation("character", "char_t1_alive") is None


def test_terminal_check_blocks_event_after_death():
    storage.save_entity(
        "character", "char_t2_dead", {"birth_year": 2000, "death_year": 2100}
    )
    _link_event("char_t2_dead", "event_t2_a", 2150)

    conflict = hard_check.check_terminal_violation("character", "char_t2_dead")

    assert conflict is not None
    assert conflict.check_type == "terminal"
    assert conflict.severity == "blocking"


def test_terminal_check_blocks_when_death_year_set_before_existing_max_event():
    storage.save_entity("character", "char_t3_dead", {"birth_year": 2000})
    _link_event("char_t3_dead", "event_t3_a", 2100)
    _link_event("char_t3_dead", "event_t3_b", 2200)

    storage.save_entity("character", "char_t3_dead", {"death_year": 2150})

    conflict = hard_check.check_terminal_violation("character", "char_t3_dead")

    assert conflict is not None
    assert conflict.severity == "blocking"


def test_lifespan_check_warns_with_birth_and_death_year():
    storage.save_entity("race", "race_t4_short", {"lifespan": 100})
    storage.save_entity(
        "character",
        "char_t4_long",
        {"birth_year": 2000, "death_year": 2300, "race": "race_t4_short"},
    )

    conflict = hard_check.check_lifespan_violation("char_t4_long")

    assert conflict is not None
    assert conflict.severity == "warning"
    assert "300" in conflict.reason


def test_lifespan_check_uses_event_years_when_death_year_missing():
    storage.save_entity("race", "race_t5_short", {"lifespan": 100})
    storage.save_entity(
        "character",
        "char_t5_long",
        {"birth_year": 2000, "race": "race_t5_short"},
    )
    _link_event("char_t5_long", "event_t5_a", 3000)

    conflict = hard_check.check_lifespan_violation("char_t5_long")

    assert conflict is not None
    assert conflict.severity == "warning"
    assert "1000" in conflict.reason


def test_lifespan_check_skipped_when_ack_true():
    storage.save_entity("race", "race_t6_short", {"lifespan": 100})
    storage.save_entity(
        "character",
        "char_t6_long",
        {"birth_year": 2000, "race": "race_t6_short"},
    )
    _link_event("char_t6_long", "event_t6_a", 3000)

    assert hard_check.check_lifespan_violation("char_t6_long") is not None

    storage.save_entity("character", "char_t6_long", {"lifespan_check_ack": True})

    assert hard_check.check_lifespan_violation("char_t6_long") is None


def test_lifespan_check_skips_when_race_or_lifespan_missing():
    storage.save_entity(
        "character", "char_t7_norace", {"birth_year": 2000, "death_year": 3000}
    )
    assert hard_check.check_lifespan_violation("char_t7_norace") is None

    storage.save_entity("race", "race_t7_unknown", {"lifespan": None})
    storage.save_entity(
        "character",
        "char_t7_unknownrace",
        {"birth_year": 2000, "death_year": 3000, "race": "race_t7_unknown"},
    )
    assert hard_check.check_lifespan_violation("char_t7_unknownrace") is None


def test_terminal_check_catches_artifact_reappearing_after_destroyed_year():
    """artifact only has lifecycle_end (destroyed_year), no lifecycle_start field at all."""
    storage.save_entity(
        "artifact",
        "item_t8_relic",
        {"current_status": "destroyed", "destroyed_year": 2100},
    )
    _link_event("item_t8_relic", "event_t8_reappear", 2150)

    conflict = hard_check.check_terminal_violation("artifact", "item_t8_relic")

    assert conflict is not None
    assert conflict.severity == "blocking"


def test_run_hard_checks_on_artifact_only_runs_terminal_check():
    storage.save_entity(
        "artifact",
        "item_t9_relic",
        {"current_status": "destroyed", "destroyed_year": 2100},
    )
    _link_event("item_t9_relic", "event_t9_reappear", 2150)

    conflicts = hard_check.run_hard_checks("artifact", "item_t9_relic")

    assert len(conflicts) == 1
    assert conflicts[0].check_type == "terminal"
    assert conflicts[0].severity == "blocking"


def test_run_hard_checks_on_location_does_not_crash_and_skips_lifespan():
    storage.save_entity(
        "location",
        "loc_t10_ruin",
        {"category": "폐허", "founded_year": 1000, "destroyed_year": 1200},
    )
    _link_event("loc_t10_ruin", "event_t10_late", 1300)

    conflicts = hard_check.run_hard_checks("location", "loc_t10_ruin")

    assert len(conflicts) == 1
    assert conflicts[0].check_type == "terminal"


def test_run_hard_checks_on_faction_returns_empty_when_no_violations():
    storage.save_entity(
        "faction", "faction_t11_calm", {"category": "왕국", "founded_year": 1000}
    )

    assert hard_check.run_hard_checks("faction", "faction_t11_calm") == []
