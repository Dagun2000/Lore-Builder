"""Phase 8 — pipeline state machine (real LLM calls, same as Phase 3/5's
suites — entity resolution setups below are deliberately exact-match so LLM
non-determinism stays scoped to inference/rag_check judgment calls, mirroring
test_e2e.py's own approach).

Section 1 re-runs test_e2e.py's 10 scenarios through main.run_pipeline_interactive
(cli_loop's state-machine path) instead of the old blocking main.run_pipeline,
as a regression check that nothing changed user-visibly. main.run_pipeline
itself is untouched and still used directly by test_e2e.py.

Section 2 exercises pipeline_session.start_session/resume_session directly
for the specific state-machine behaviors Phase 8 introduces (pausing,
resuming, one-decision-at-a-time diff review, immediate abort on a blocking
conflict).
"""

from src import main, pipeline_session, storage


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
# 1. test_e2e.py's 10 scenarios, re-run through the state machine.
# ---------------------------------------------------------------------------

def test_new_character_death_event_saves_normally(monkeypatch):
    monkeypatch.setattr(
        main, "_prompt", _script(("기본값", ""), ("사망", "예"), ("승인하시겠습니까", "y"))
    )

    result = main.run_pipeline_interactive("2200년, [로한]이 몬스터와 싸우다가 죽었다.")

    assert result["status"] == "saved"
    entity_id = result["resolved_entities"]["로한"]
    assert entity_id.startswith("char_")
    entity = storage.get_entity("character", entity_id)
    assert entity["name"] == "로한"
    assert entity["death_year"] == 2200


def test_already_dead_character_reappearing_is_rejected(monkeypatch):
    storage.save_entity(
        "character",
        "char_p8_deadman",
        {
            "name": "마르쿠스",
            "birth_year": 2000,
            "death_year": 2100,
            "notes": "마르쿠스는 2100년에 사망한 것으로 기록된 인물이다.",
        },
    )
    monkeypatch.setattr(main, "_prompt", _script())  # no prompts should be reachable

    result = main.run_pipeline_interactive("2150년, [마르쿠스]가 마을 광장에 나타났다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "hard_check"
    assert any(c.severity == "blocking" for c in result["conflicts"])


def test_death_year_out_of_order_with_existing_events_is_rejected(monkeypatch):
    storage.save_entity(
        "character",
        "char_p8_reverse",
        {
            "name": "델피나",
            "birth_year": 2000,
            "death_year": 2150,
            "notes": "델피나는 시간 순서가 꼬인 기록으로 유명한 인물이다.",
        },
    )
    storage.save_entity("timeline", "event_p8_reverse_a", {"year": 2100})
    storage.add_event_pointer("char_p8_reverse", "event_p8_reverse_a")
    storage.save_entity("timeline", "event_p8_reverse_b", {"year": 2200})
    storage.add_event_pointer("char_p8_reverse", "event_p8_reverse_b")
    monkeypatch.setattr(main, "_prompt", _script())

    result = main.run_pipeline_interactive("2180년, [델피나]가 시장에서 물건을 샀다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "hard_check"
    assert any(c.check_type == "terminal" and c.severity == "blocking" for c in result["conflicts"])


def test_lifespan_warning_accept_sets_ack_and_suppresses_future_popup(monkeypatch):
    storage.save_entity("race", "race_p8_short", {"lifespan": 50, "notes": "수명이 매우 짧은 종족."})
    storage.save_entity(
        "character",
        "char_p8_longlived",
        {
            "name": "세라핀",
            "birth_year": 2000,
            "race": "race_p8_short",
            "notes": "세라핀은 수명 논란이 있는 인물이다.",
        },
    )
    monkeypatch.setattr(
        main, "_prompt", _script(("그래도 저장", "그래도 저장"), ("승인하시겠습니까", "y"))
    )

    result1 = main.run_pipeline_interactive("2100년, [세라핀]이 시장에 나타났다.")
    assert any(c.check_type == "lifespan" for c in result1["conflicts"])
    assert result1["status"] == "saved"
    assert storage.get_entity("character", "char_p8_longlived")["lifespan_check_ack"]

    result2 = main.run_pipeline_interactive("2200년, [세라핀]이 다시 나타났다.")
    assert not any(c.check_type == "lifespan" for c in result2["conflicts"])


def test_lifespan_warning_computed_from_event_years_only(monkeypatch):
    storage.save_entity("race", "race_p8_short2", {"lifespan": 50, "notes": "수명이 짧은 종족 2."})
    storage.save_entity(
        "character",
        "char_p8_nodates",
        {
            "name": "유리안",
            "race": "race_p8_short2",
            "notes": "유리안은 생몰년이 기록되지 않은 인물이다.",
        },
    )
    storage.save_entity("timeline", "event_p8_nodates_a", {"year": 2000})
    storage.add_event_pointer("char_p8_nodates", "event_p8_nodates_a")
    monkeypatch.setattr(
        main, "_prompt", _script(("그래도 저장", "그래도 저장"), ("승인하시겠습니까", "y"))
    )

    result = main.run_pipeline_interactive("2100년, [유리안]이 시장에 나타났다.")

    lifespan_conflicts = [c for c in result["conflicts"] if c.check_type == "lifespan"]
    assert lifespan_conflicts
    assert "100" in lifespan_conflicts[0].reason


def test_destroyed_artifact_reappearing_is_rejected(monkeypatch):
    storage.save_entity(
        "artifact",
        "item_p8_relic",
        {
            "name": "오브시디언",
            "current_status": "destroyed",
            "destroyed_year": 2100,
            "notes": "오브시디언은 2100년에 파괴된 것으로 기록된 유물이다.",
        },
    )
    monkeypatch.setattr(main, "_prompt", _script())

    result = main.run_pipeline_interactive("2150년, [오브시디언]을 누군가 손에 들고 나타났다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "hard_check"
    assert any(c.severity == "blocking" for c in result["conflicts"])


def test_escape_clears_imprisoned_status_automatically(monkeypatch):
    storage.save_entity(
        "character",
        "char_p8_prisoner",
        {"name": "브락스", "birth_year": 2000, "notes": "브락스는 용병 출신의 인물이다."},
    )
    storage.save_entity(
        "timeline",
        "event_p8_prisoner_status",
        {
            "entity": "char_p8_prisoner",
            "predicate": "imprisoned",
            "target": None,
            "start_year": 2050,
            "end_year": None,
            "notes": "브락스가 2050년부터 수감 상태다.",
        },
    )
    storage.add_event_pointer("char_p8_prisoner", "event_p8_prisoner_status")

    monkeypatch.setattr(
        main, "_prompt", _script(("그래도 저장하시겠습니까", "그래도 저장"), ("승인하시겠습니까", "y"))
    )

    result = main.run_pipeline_interactive("2100년, [브락스]가 탈출해서 마을로 도망쳤다.")

    assert result["status"] == "saved"
    assert not storage.get_current_state("char_p8_prisoner", "imprisoned")


def test_imprisoned_character_fighting_raises_conflict_popup(monkeypatch):
    storage.save_entity(
        "character",
        "char_p8_prisoner2",
        {"name": "카인", "birth_year": 2000, "notes": "카인은 용병 출신의 인물이다."},
    )
    storage.save_entity(
        "timeline",
        "event_p8_prisoner2_status",
        {
            "entity": "char_p8_prisoner2",
            "predicate": "imprisoned",
            "target": None,
            "start_year": 2050,
            "end_year": None,
            "notes": "카인이 2050년부터 수감 상태다.",
        },
    )
    storage.add_event_pointer("char_p8_prisoner2", "event_p8_prisoner2_status")

    responder = _script(("그래도 저장", "그래도 저장"), ("승인하시겠습니까", "y"))
    monkeypatch.setattr(main, "_prompt", responder)

    result = main.run_pipeline_interactive("2100년, [카인]이 수감 중에 전장에서 검을 휘둘렀다.")

    conflict_prompts = [m for m in responder.calls if "그래도 저장하시겠습니까" in m]
    assert conflict_prompts  # the popup was actually shown
    assert any(j.type == "conflict" for j in result.get("rag_judgments", []))
    assert result["status"] == "saved"
    assert storage.get_current_state("char_p8_prisoner2", "imprisoned")


def test_elf_eating_meat_triggers_notes_conflict(monkeypatch):
    monkeypatch.setattr(main, "_prompt", _script(("그래도 저장하시겠습니까", "취소")))

    result = main.run_pipeline_interactive("2100년, [미라]가 사냥한 고기를 먹었다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "rag_check"
    assert any(j.type == "notes_conflict" for j in result["rag_judgments"])


def test_magic_without_mana_stone_violates_world_rule(monkeypatch):
    monkeypatch.setattr(main, "_prompt", _script(("그래도 저장하시겠습니까", "취소")))

    result = main.run_pipeline_interactive("2100년, 손끝에서 불꽃을 만들어냈다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "rag_check"
    assert any(j.type == "rule_violation" for j in result["rag_judgments"])


# ---------------------------------------------------------------------------
# 2. State-machine-specific behavior, driven directly via pipeline_session.
# ---------------------------------------------------------------------------

def test_new_entity_pauses_for_category_and_name_before_anything_else():
    session = pipeline_session.start_session("2100년, [파블로]가 몬스터와 싸우다가 죽었다.")

    assert session.pending_decision.decision_type == "entity_category_and_name"
    assert session.pending_decision.payload["tag"] == "파블로"
    assert session.pending_decision.payload["inferred_category"] == "character"
    assert "character" in session.pending_decision.payload["categories"]
    assert session.pending_decision.payload["has_name_field"] is True


def test_category_and_name_edit_advances_to_terminal_status_for_character():
    session = pipeline_session.start_session("2100년, [파블로]가 몬스터와 싸우다가 죽었다.")
    session = pipeline_session.resume_session(
        session.session_id, {"category": None, "name": "파블로", "action": "edit"}
    )

    assert session.pending_decision.decision_type == "entity_terminal_status"
    assert session.pending_decision.payload["tag"] == "파블로"


def test_category_override_changes_the_field_set_offered_next():
    # Patch A completion criterion 2: switching the category in the combobox
    # must recompute the field set for the *new* category, not the LLM's.
    session = pipeline_session.start_session("2100년, [파블로]가 몬스터와 싸우다가 죽었다.")
    assert session.pending_decision.payload["inferred_category"] == "character"

    session = pipeline_session.resume_session(
        session.session_id, {"category": "faction", "name": "파블로", "action": "edit"}
    )

    assert session.pending_decision.decision_type == "entity_required_field"
    assert session.pending_decision.payload["category"] == "faction"
    field_names = {f["name"] for f in session.pending_decision.payload["fields"]}
    assert "category" in field_names  # faction.category, required
    assert "birth_year" not in field_names  # character-only field must be gone


def test_save_and_continue_leaves_remaining_fields_blank():
    # Patch A completion criterion 3.
    session = pipeline_session.start_session("2100년, [그림자단]이라는 새로운 세력이 등장했다.")
    assert session.pending_decision.decision_type == "entity_category_and_name"
    assert session.pending_decision.payload["inferred_category"] == "faction"

    session = pipeline_session.resume_session(
        session.session_id, {"category": None, "name": "그림자단", "action": "save"}
    )

    while session.pending_decision is not None:
        dt = session.pending_decision.decision_type
        assert dt != "entity_required_field"  # fast path must never ask for more
        response = "그래도 저장" if dt in ("hard_check_warning", "rag_judgment") else True
        session = pipeline_session.resume_session(session.session_id, response)

    assert session.stage == "done"
    entity_id = session.result["resolved_entities"]["그림자단"]
    entity = storage.get_entity("faction", entity_id)
    assert entity["name"] == "그림자단"
    assert entity.get("category") is None


def test_edit_reveals_full_field_form_for_non_character():
    # Patch A completion criterion 4 (required forced, optional included).
    session = pipeline_session.start_session("2100년, [블랙로터스]라는 새로운 세력이 등장했다.")
    assert session.pending_decision.decision_type == "entity_category_and_name"

    session = pipeline_session.resume_session(
        session.session_id, {"category": None, "name": "블랙로터스", "action": "edit"}
    )

    assert session.pending_decision.decision_type == "entity_required_field"
    assert session.pending_decision.payload["category"] == "faction"
    fields_by_name = {f["name"]: f for f in session.pending_decision.payload["fields"]}
    assert fields_by_name["category"]["required"] is True
    assert "founded_year" in fields_by_name  # optional, still offered in "edit"
    assert fields_by_name["founded_year"]["required"] is False


def test_cancel_entity_creation_aborts_whole_input():
    session = pipeline_session.start_session("2100년, [무명인]이 시장에 나타났다.")
    assert session.pending_decision.decision_type == "entity_category_and_name"

    session = pipeline_session.resume_session(
        session.session_id, {"category": None, "name": "무명인", "action": "cancel"}
    )

    assert session.pending_decision is None
    assert session.stage == "aborted"
    assert session.result["status"] == "cancelled"


def test_duplicate_exact_name_pauses_for_candidate_selection():
    storage.save_entity("character", "char_p8_dup_a", {"name": "동명"})
    storage.save_entity("character", "char_p8_dup_b", {"name": "동명"})

    session = pipeline_session.start_session("2100년, [동명]이 시장에 나타났다.")

    assert session.pending_decision.decision_type == "entity_candidates"
    assert set(session.pending_decision.payload["candidates"]) == {"char_p8_dup_a", "char_p8_dup_b"}
    assert session.pending_decision.payload["allow_create"] is False


def test_hard_check_warning_accept_persists_lifespan_ack():
    storage.save_entity("race", "race_p8_ack", {"lifespan": 50})
    storage.save_entity(
        "character",
        "char_p8_ack",
        {"name": "그레타", "birth_year": 2000, "race": "race_p8_ack", "notes": "그레타는 수명 논란이 있는 인물이다."},
    )

    session = pipeline_session.start_session("2100년, [그레타]가 시장에 나타났다.")

    assert session.pending_decision.decision_type == "hard_check_warning"
    assert session.pending_decision.payload["check_type"] == "lifespan"

    session = pipeline_session.resume_session(session.session_id, "그래도 저장")

    while session.pending_decision is not None:
        dt = session.pending_decision.decision_type
        response = "그래도 저장" if dt == "rag_judgment" else True
        session = pipeline_session.resume_session(session.session_id, response)

    assert session.stage == "done"
    assert storage.get_entity("character", "char_p8_ack")["lifespan_check_ack"]


def test_diff_items_are_resumed_one_at_a_time():
    session = pipeline_session.start_session("2100년, [로라]와 [단테]가 동맹을 맺었다.")

    # Drive past whatever entity/hard-check/rag decisions come up on the way
    # to the diff review stage — this scenario's point is the diff loop, not
    # entity resolution, so accept every intermediate decision plainly.
    while session.pending_decision is not None and session.pending_decision.decision_type != "diff_item":
        dt = session.pending_decision.decision_type
        if dt == "entity_candidates":
            response = session.pending_decision.payload["candidates"][0]
        elif dt == "entity_category_and_name":
            response = {"category": None, "name": None, "action": "edit"}
        elif dt == "entity_terminal_status":
            response = "아니오"
        elif dt == "entity_required_field":
            response = {
                f["name"]: "미상"
                for f in session.pending_decision.payload["fields"]
                if f["required"]
            }
        else:  # hard_check_warning / rag_judgment
            response = "그래도 저장"
        session = pipeline_session.resume_session(session.session_id, response)

    assert session.pending_decision is not None
    assert session.pending_decision.decision_type == "diff_item"
    total = len(session.diff)
    assert total >= 2  # timeline record + at least one relationship

    seen = 0
    while session.pending_decision is not None:
        assert session.pending_decision.decision_type == "diff_item"
        assert session.stage == "reviewing_diff"
        seen += 1
        session = pipeline_session.resume_session(session.session_id, True)
        if seen < total:
            assert session.pending_decision is not None  # not finished after just one

    assert seen == total
    assert session.stage == "done"


def test_completed_saved_session_result_matches_run_pipeline_shape():
    session = pipeline_session.start_session("2200년, [헥터]가 몬스터와 싸우다가 죽었다.")

    while session.pending_decision is not None:
        dt = session.pending_decision.decision_type
        if dt == "entity_category_and_name":
            response = {"category": None, "name": None, "action": "edit"}
        elif dt == "entity_terminal_status":
            response = "예"
        elif dt == "entity_required_field":
            response = {
                f["name"]: "미상"
                for f in session.pending_decision.payload["fields"]
                if f["required"]
            }
        elif dt in ("hard_check_warning", "rag_judgment"):
            response = "그래도 저장"
        else:  # diff_item
            response = True
        session = pipeline_session.resume_session(session.session_id, response)

    assert session.stage == "done"
    expected_keys = {
        "status", "parsed", "resolved_entities", "inferred_event",
        "rag_judgments", "conflicts", "diff", "approved", "applied",
    }
    assert expected_keys <= set(session.result.keys())
    assert session.result["status"] == "saved"


def test_blocking_hard_check_aborts_immediately_without_decision():
    storage.save_entity(
        "character",
        "char_p8_blocked",
        {
            "name": "올리비아",
            "birth_year": 2000,
            "death_year": 2100,
            "notes": "올리비아는 2100년에 사망한 것으로 기록된 인물이다.",
        },
    )

    session = pipeline_session.start_session("2150년, [올리비아]가 마을 광장에 나타났다.")

    assert session.pending_decision is None
    assert session.stage == "aborted"
    assert session.result["status"] == "rejected"
    assert any(c.severity == "blocking" for c in session.result["conflicts"])
