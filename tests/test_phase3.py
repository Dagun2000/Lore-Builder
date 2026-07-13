"""Phase 3 tests call the real reasoning-tier LLM (per spec, unlike Phase 2's
mocked tests) — OPENAI_API_KEY must be set in the environment. Assertions
check the *direction* of each judgment (None vs not, .type), not exact text,
since LLM output isn't 100% deterministic.
"""

from src import inference, rag_check


def test_infer_relationship_and_event_basic():
    resolved = {"쟝": "char_jang", "검은 산양 여관": "loc_black_goat_inn"}

    result = inference.infer_relationship_and_event(
        resolved, "쟝이 검은 산양 여관에서 얻어맞았다.", 2100
    )

    assert result.event_summary
    assert any(
        r.get("subject") in resolved.values() and r.get("object") in resolved.values()
        for r in result.relationships
    )
    assert result.status_effect is None


def test_check_status_consistency_detects_clear():
    judgment = rag_check.check_status_consistency(
        "char_jang", "쟝이 탈출해서 마을로 도망쳤다."
    )

    assert judgment is not None
    assert judgment.type == "clears_status"


def test_check_status_consistency_detects_conflict():
    judgment = rag_check.check_status_consistency(
        "char_jang", "쟝이 수감 중에 전장에서 검을 휘둘렀다."
    )

    assert judgment is not None
    assert judgment.type == "conflict"


def test_check_rule_violation_detects_magic_without_mana_stone():
    hard_rule_docs = rag_check._get_hard_rule_texts()

    judgment = rag_check.check_rule_violation("손끝에서 불꽃을 만들어냈다.", hard_rule_docs)

    assert judgment is not None
    assert judgment.type == "rule_violation"


def test_check_notes_conflict_detects_elf_eating_meat():
    # char_mira is race_elf, whose notes say it doesn't eat meat.
    judgment = rag_check.check_notes_conflict(
        ["char_mira"], "미라가 사냥한 고기를 먹었다."
    )

    assert judgment is not None
    assert judgment.type == "notes_conflict"


def test_no_false_positives_for_mundane_event():
    # Uses char_mira (no active status_effect) rather than char_jang, since
    # char_jang carries an unresolved "imprisoned" status from the seed data
    # used by the two tests above — that would make "took a walk" a genuinely
    # ambiguous case for check_status_consistency, not a clean no-op baseline.
    raw_text = "미라가 마을을 산책했다."
    context_docs = rag_check.retrieve_context(["char_mira"], raw_text)
    hard_rule_docs = rag_check._get_hard_rule_texts()
    combined_docs = list(dict.fromkeys(context_docs + hard_rule_docs))

    assert rag_check.check_rule_violation(raw_text, combined_docs) is None
    assert rag_check.check_notes_conflict(["char_mira"], raw_text) is None
    assert rag_check.check_status_consistency("char_mira", raw_text) is None
