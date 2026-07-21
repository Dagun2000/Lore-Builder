"""Phase 3 tests call the real reasoning-tier LLM (per spec, unlike Phase 2's
mocked tests) — OPENAI_API_KEY must be set in the environment. Assertions
check the *direction* of each judgment (None vs not, .type), not exact text,
since LLM output isn't 100% deterministic.
"""

from src import inference, rag_check


def test_infer_event_basic_point_event():
    resolved = {"데이비드": "char_데이비드", "검은 산양 여관": "loc_검은_염소_주점"}

    result = inference.infer_event(
        resolved, "데이비드가 검은 산양 여관에서 얻어맞았다.", [2100]
    )

    assert result.is_single_event
    assert result.event_type == "point"
    assert result.event_summary
    assert set(result.involved_entities) <= set(resolved.values())
    assert result.duration_effect is None


def test_check_status_consistency_detects_clear():
    # char_데이비드's seeded "imprisoned" range starts 2085, still open — 2100
    # falls inside it, so this is a genuine gated-in case.
    judgments = rag_check.check_status_consistency(
        ["char_데이비드"], "데이비드가 탈출해서 마을로 도망쳤다.", 2100
    )

    assert any(j.type == "clears_status" for j in judgments)


def test_check_status_consistency_detects_conflict():
    judgments = rag_check.check_status_consistency(
        ["char_데이비드"], "데이비드가 수감 중에 전장에서 검을 휘둘렀다.", 2100
    )

    assert any(j.type == "conflict" for j in judgments)


def test_check_rule_violation_detects_magic_without_mana_stone():
    hard_rule_docs = rag_check._get_hard_rule_texts()

    judgment = rag_check.check_rule_violation([], "손끝에서 불꽃을 만들어냈다.", hard_rule_docs)

    assert judgment is not None
    assert judgment.type == "rule_violation"


def test_check_notes_conflict_detects_elf_eating_meat():
    # char_미라 is race_엘프, whose notes say it doesn't eat meat.
    judgment = rag_check.check_notes_conflict(
        ["char_미라"], "미라가 사냥한 고기를 먹었다."
    )

    assert judgment is not None
    assert judgment.type == "notes_conflict"


def test_check_rule_and_notes_combined_still_detects_notes_conflict():
    # Phase 10 patch 18: check_rule_violation + check_notes_conflict merged
    # into one LLM call to cut redundant context-resending. Same fixture as
    # test_check_notes_conflict_detects_elf_eating_meat, run through the
    # combined entry point instead, to confirm the merge doesn't lose either
    # judgment type in the process.
    hard_rule_docs = rag_check._get_hard_rule_texts()
    judgments = rag_check.check_rule_and_notes(
        ["char_미라"], "미라가 사냥한 고기를 먹었다.", hard_rule_docs
    )

    assert any(j.type == "notes_conflict" for j in judgments)


def test_no_false_positives_for_mundane_event():
    # Uses char_미라 (no active status_effect) rather than char_데이비드, since
    # char_데이비드 carries an unresolved "imprisoned" status from the seed data
    # used by the two tests above — that would make "took a walk" a genuinely
    # ambiguous case for check_status_consistency, not a clean no-op baseline.
    raw_text = "미라가 마을을 산책했다."
    context_docs = rag_check.retrieve_context(["char_미라"], raw_text)
    hard_rule_docs = rag_check._get_hard_rule_texts()
    combined_docs = list(dict.fromkeys(context_docs + hard_rule_docs))

    assert rag_check.check_rule_violation(["char_미라"], raw_text, combined_docs) is None
    assert rag_check.check_notes_conflict(["char_미라"], raw_text) is None
    assert rag_check.check_status_consistency(["char_미라"], raw_text, 2100) == []
