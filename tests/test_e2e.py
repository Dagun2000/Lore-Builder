"""End-to-end pipeline tests — Phase 5 (real LLM calls, per spec, not mocked).

Each scenario pre-seeds dedicated test entities directly via storage (rather
than routing setup through mapping.resolve_entity's interactive creation
flow), so entity *resolution* during the test is a deterministic
exact/partial string match with zero prompts — the only prompts scripted
are the ones the scenario is actually about. This keeps LLM non-determinism
scoped to inference/rag_check judgment calls, not to unrelated setup steps.

_script() below answers _prompt() calls by message substring and raises
loudly if a scenario asks more questions than expected, instead of silently
looping forever on an unrecognized prompt.
"""

from src import main, mapping, approval, storage


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


def _patch_prompts(monkeypatch, responder):
    monkeypatch.setattr(mapping, "_prompt", responder)
    monkeypatch.setattr(approval, "_prompt", responder)


# ---------------------------------------------------------------------------
# 1. New character, dies in a new event -> saved normally
# ---------------------------------------------------------------------------


def test_new_character_death_event_saves_normally(monkeypatch):
    responder = _script(("사망", "예"), ("승인하시겠습니까", "y"))
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2200년, [가론]이 몬스터와 싸우다가 죽었다.")

    assert result["status"] == "saved"
    entity_id = result["resolved_entities"]["가론"]
    assert entity_id.startswith("char_")
    entity = storage.get_entity("character", entity_id)
    assert entity["death_year"] == 2200


# ---------------------------------------------------------------------------
# 2. Already-dead character reappears -> immediate blocking rejection
# ---------------------------------------------------------------------------


def test_already_dead_character_reappearing_is_rejected(monkeypatch):
    storage.save_entity(
        "character",
        "char_e2e_deadman",
        {
            "birth_year": 2000,
            "death_year": 2100,
            "notes": "달튼은 2100년에 사망한 것으로 기록된 인물이다.",
        },
    )
    responder = _script()  # no prompts should even be reachable here
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2150년, [달튼]이 마을 광장에 나타났다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "hard_check"
    assert any(c.severity == "blocking" for c in result["conflicts"])


# ---------------------------------------------------------------------------
# 3. Death year inconsistent with already-recorded events -> rejected
# ---------------------------------------------------------------------------


def test_death_year_out_of_order_with_existing_events_is_rejected(monkeypatch):
    storage.save_entity(
        "character",
        "char_e2e_reverse",
        {
            "birth_year": 2000,
            "death_year": 2150,
            "notes": "카심은 시간 순서가 꼬인 기록으로 유명한 인물이다.",
        },
    )
    storage.save_entity("timeline", "event_e2e_reverse_a", {"year": 2100})
    storage.save_entity(
        "relationship",
        "rel_e2e_reverse_a",
        {"subject": "char_e2e_reverse", "predicate": "involved_in", "object": "event_e2e_reverse_a"},
    )
    storage.save_entity("timeline", "event_e2e_reverse_b", {"year": 2200})
    storage.save_entity(
        "relationship",
        "rel_e2e_reverse_b",
        {"subject": "char_e2e_reverse", "predicate": "involved_in", "object": "event_e2e_reverse_b"},
    )
    responder = _script()
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2180년, [카심]이 시장에서 물건을 샀다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "hard_check"
    assert any(c.check_type == "terminal" and c.severity == "blocking" for c in result["conflicts"])


# ---------------------------------------------------------------------------
# 4. Lifespan exceeded -> warning popup -> "save anyway" -> ack persists
# ---------------------------------------------------------------------------


def test_lifespan_warning_accept_sets_ack_and_suppresses_future_popup(monkeypatch):
    storage.save_entity("race", "race_e2e_short", {"lifespan": 50, "notes": "수명이 매우 짧은 종족."})
    storage.save_entity(
        "character",
        "char_e2e_longlived",
        {
            "birth_year": 2000,
            "race": "race_e2e_short",
            "notes": "브란은 수명 논란이 있는 인물이다.",
        },
    )
    responder = _script(("그래도 저장", "그래도 저장"), ("승인하시겠습니까", "y"))
    _patch_prompts(monkeypatch, responder)

    result1 = main.run_pipeline("2100년, [브란]이 시장에 나타났다.")
    assert any(c.check_type == "lifespan" for c in result1["conflicts"])
    assert result1["status"] == "saved"
    assert storage.get_entity("character", "char_e2e_longlived")["lifespan_check_ack"]

    result2 = main.run_pipeline("2200년, [브란]이 다시 나타났다.")
    assert not any(c.check_type == "lifespan" for c in result2["conflicts"])


# ---------------------------------------------------------------------------
# 5. Lifespan computed purely from the event-year array (no birth/death_year)
# ---------------------------------------------------------------------------


def test_lifespan_warning_computed_from_event_years_only(monkeypatch):
    storage.save_entity("race", "race_e2e_short2", {"lifespan": 50, "notes": "수명이 짧은 종족 2."})
    storage.save_entity(
        "character",
        "char_e2e_nodates",
        {"race": "race_e2e_short2", "notes": "가웨인은 생몰년이 기록되지 않은 인물이다."},
    )
    storage.save_entity("timeline", "event_e2e_nodates_a", {"year": 2000})
    storage.save_entity(
        "relationship",
        "rel_e2e_nodates_a",
        {"subject": "char_e2e_nodates", "predicate": "involved_in", "object": "event_e2e_nodates_a"},
    )
    responder = _script(("그래도 저장", "그래도 저장"), ("승인하시겠습니까", "y"))
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2100년, [가웨인]이 시장에 나타났다.")

    lifespan_conflicts = [c for c in result["conflicts"] if c.check_type == "lifespan"]
    assert lifespan_conflicts
    assert "100" in lifespan_conflicts[0].reason  # 2000~2100 computed age


# ---------------------------------------------------------------------------
# 6. Destroyed artifact reappearing -> rejected (possession/existence conflict)
# ---------------------------------------------------------------------------


def test_destroyed_artifact_reappearing_is_rejected(monkeypatch):
    storage.save_entity(
        "artifact",
        "item_e2e_relic",
        {
            "current_status": "destroyed",
            "destroyed_year": 2100,
            "notes": "이렐릭은 2100년에 파괴된 것으로 기록된 유물이다.",
        },
    )
    responder = _script()
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2150년, [이렐릭]을 누군가 손에 들고 나타났다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "hard_check"
    assert any(c.severity == "blocking" for c in result["conflicts"])


# ---------------------------------------------------------------------------
# 7. Imprisoned character "escapes" -> status auto-cleared, no popup needed
# ---------------------------------------------------------------------------


def test_escape_clears_imprisoned_status_automatically(monkeypatch):
    storage.save_entity(
        "character",
        "char_e2e_prisoner",
        {
            "birth_year": 2000,
            "active_status_effects": ["imprisoned"],
            "notes": "가일은 용병 출신의 인물이다.",
        },
    )
    # "그래도 저장" is a defensive fallback in case some other judgment (e.g.
    # rule_violation) also fires alongside the expected clears_status one —
    # the assertion below is what actually verifies the auto-clear behavior.
    responder = _script(("그래도 저장하시겠습니까", "그래도 저장"), ("승인하시겠습니까", "y"))
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2100년, [가일]이 탈출해서 마을로 도망쳤다.")

    assert result["status"] == "saved"
    assert "imprisoned" not in storage.get_status_effects("char_e2e_prisoner")


# ---------------------------------------------------------------------------
# 8. Imprisoned character "fights on a battlefield" -> conflict popup
# ---------------------------------------------------------------------------


def test_imprisoned_character_fighting_raises_conflict_popup(monkeypatch):
    storage.save_entity(
        "character",
        "char_e2e_prisoner2",
        {
            "birth_year": 2000,
            "active_status_effects": ["imprisoned"],
            "notes": "테오는 용병 출신의 인물이다.",
        },
    )
    responder = _script(("그래도 저장", "그래도 저장"), ("승인하시겠습니까", "y"))
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2100년, [테오]가 수감 중에 전장에서 검을 휘둘렀다.")

    conflict_prompts = [m for m in responder.calls if "conflict" in m or "그래도 저장하시겠습니까" in m]
    assert conflict_prompts  # the popup was actually shown
    assert any(j.type == "conflict" for j in result.get("rag_judgments", []))
    assert result["status"] == "saved"
    # Still marked imprisoned — the contradiction was accepted, not resolved.
    assert "imprisoned" in storage.get_status_effects("char_e2e_prisoner2")


# ---------------------------------------------------------------------------
# 9. Elf character (notes: doesn't eat meat) eats meat -> notes conflict
# ---------------------------------------------------------------------------


def test_elf_eating_meat_triggers_notes_conflict(monkeypatch):
    responder = _script(("그래도 저장하시겠습니까", "취소"))
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2100년, [미라]가 사냥한 고기를 먹었다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "rag_check"
    assert any(j.type == "notes_conflict" for j in result["rag_judgments"])


# ---------------------------------------------------------------------------
# 10. World rule (hard_rule) violation -> caught at Step 4
# ---------------------------------------------------------------------------


def test_magic_without_mana_stone_violates_world_rule(monkeypatch):
    responder = _script(("그래도 저장하시겠습니까", "취소"))
    _patch_prompts(monkeypatch, responder)

    result = main.run_pipeline("2100년, 손끝에서 불꽃을 만들어냈다.")

    assert result["status"] == "rejected"
    assert result["stage"] == "rag_check"
    assert any(j.type == "rule_violation" for j in result["rag_judgments"])
