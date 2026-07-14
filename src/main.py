"""Full pipeline orchestration — Phase 5.

Wires together every prior phase's piece (schema, storage, hard_check,
parser, mapping, inference, rag_check, archivist, approval) into one CLI
loop. run_pipeline returns a structured result dict (not just prints), so a
future GUI can render outcomes without scraping stdout — see approval.py's
docstring for the note on _prompt/_print being the swap point for that GUI.
"""

if __package__:
    from . import (
        archivist,
        approval,
        hard_check,
        inference,
        mapping,
        parser,
        pipeline_session,
        rag_check,
        schema,
        storage,
    )
else:  # allows `python src/main.py` to run directly, not just `python -m src.main`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import (
        archivist,
        approval,
        hard_check,
        inference,
        mapping,
        parser,
        pipeline_session,
        rag_check,
        schema,
        storage,
    )


def _prompt(message: str) -> str:
    return input(message)


def _print(message: str = "") -> None:
    print(message)


def _apply_diff(approved: list) -> list:
    creates = [c for c in approved if c.action == "create"]
    updates = [c for c in approved if c.action == "update"]

    applied = []
    for item in creates + updates:
        storage.save_entity(item.category, item.entity_id, item.fields)
        if item.body:
            storage.save_to_chroma(
                item.entity_id, item.body, {"category": item.category}
            )
        applied.append(item)
    return applied


def _describe_applied(item) -> str:
    return f"{item.entity_id}(갱신)" if item.action == "update" else item.entity_id


def run_pipeline(user_input: str) -> dict:
    try:
        parsed = parser.parse_input(user_input)
    except ValueError as exc:
        _print(f"입력 오류: {exc}")
        return {"status": "error", "stage": "parse", "message": str(exc)}

    resolved_entities = {}
    for tag in parsed.tags:
        resolved_entities[tag] = mapping.resolve_entity(
            tag, parsed.raw_text, parsed.year
        )

    inferred_event = inference.infer_relationship_and_event(
        resolved_entities, parsed.raw_text, parsed.year
    )

    rag_judgments = rag_check.run_rag_checks(
        list(resolved_entities.values()), parsed.raw_text
    )

    conflicts = []
    for entity_id in resolved_entities.values():
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        conflicts.extend(
            hard_check.run_hard_checks(category, entity_id, extra_years=[parsed.year])
        )

    if not approval.review_hard_check_conflicts(conflicts):
        _print("하드체크 결과에 따라 저장이 중단되었습니다.")
        return {
            "status": "rejected",
            "stage": "hard_check",
            "resolved_entities": resolved_entities,
            "conflicts": conflicts,
        }

    if not approval.review_rag_judgments(rag_judgments):
        _print("RAG 검증 결과에 따라 저장이 중단되었습니다.")
        return {
            "status": "rejected",
            "stage": "rag_check",
            "resolved_entities": resolved_entities,
            "rag_judgments": rag_judgments,
        }

    diff = archivist.build_diff(parsed, resolved_entities, inferred_event, rag_judgments)
    approved = approval.review_diff(diff)

    if not approved:
        _print("승인된 변경사항이 없어 저장할 내용이 없습니다.")
        return {
            "status": "no_changes",
            "resolved_entities": resolved_entities,
            "diff": diff,
            "approved": [],
        }

    applied = _apply_diff(approved)
    _print("저장 완료: " + ", ".join(_describe_applied(c) for c in applied))

    return {
        "status": "saved",
        "parsed": parsed,
        "resolved_entities": resolved_entities,
        "inferred_event": inferred_event,
        "rag_judgments": rag_judgments,
        "conflicts": conflicts,
        "diff": diff,
        "approved": approved,
        "applied": applied,
    }


# ---------------------------------------------------------------------------
# Phase 8 — cli_loop rendered on top of pipeline_session's state machine.
#
# run_pipeline above stays exactly as it was (still calls mapping.py/
# approval.py's blocking input() functions directly) so nothing that already
# depends on it breaks. cli_loop no longer calls it, though: it now drives
# pipeline_session.start_session/resume_session, rendering each
# PendingDecision as the identical prompt text the old blocking functions
# used to print — same user-visible CLI, state machine underneath. A future
# GUI renders the same PendingDecision.payload as widgets instead of prompts.
# ---------------------------------------------------------------------------

def _render_entity_candidates(payload: dict) -> str:
    tag = payload["tag"]
    candidates = payload["candidates"]
    allow_create = payload.get("allow_create", False)

    if allow_create:
        _print(f"[{tag}]와(과) 정확히 일치하는 항목은 없지만, 비슷한 후보가 있습니다:")
    else:
        _print(f"[{tag}]와(과) 일치하는 후보가 여러 개입니다:")
    for i, entity_id in enumerate(candidates, start=1):
        _print(f"  {i}. {entity_id}")
    create_choice = len(candidates) + 1
    if allow_create:
        _print(f"  {create_choice}. 새로 작성 (신규 엔티티 생성)")

    upper = create_choice if allow_create else len(candidates)
    while True:
        choice = _prompt(f"번호를 선택하세요 (1-{upper}): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
            if allow_create and idx == create_choice:
                return pipeline_session.CREATE_NEW
        _print("잘못된 번호입니다.")


def _render_entity_name(payload: dict) -> str:
    return _prompt(
        f'이름 (기본값: "{payload["tag"]}", Enter로 그대로 사용, 다른 값 입력 시 변경): '
    ).strip()


def _render_entity_terminal_status(payload: dict):
    tag, year = payload["tag"], payload["year"]
    answer = _prompt(
        f"[{tag}]가 이 사건({year}년)으로 사망(또는 활동 종료)한 것으로 "
        f"추정됩니다. death_year={year}로 저장할까요? [예/아니오/수정]: "
    ).strip()
    if answer == "수정":
        value = _prompt("새로운 death_year 값 (비우면 설정 안 함): ").strip()
        return {"수정": ({"death_year": int(value)} if value else {})}
    return answer


def _render_entity_required_field(payload: dict) -> dict:
    fields = payload["fields"]
    names = ", ".join(f["name"] for f in fields)
    _print(f"[{payload['category']}] 필수 필드: {names}")

    response = {}
    for f in fields:
        while True:
            value = _prompt(f"{f['name']} 값 입력 (필수): ").strip()
            if value:
                response[f["name"]] = value
                break
            _print("필수 필드는 비워둘 수 없습니다.")
    return response


def _render_hard_check_warning(payload: dict) -> str:
    _print(f"[경고] {payload['entity_id']}: {payload['reason']}")
    return _prompt("그래도 저장하시겠습니까? [그래도 저장/수정/취소]: ").strip()


def _render_rag_judgment(payload: dict) -> str:
    _print(f"[{payload['judgment_type']}] {payload['reason']}")
    return _prompt("그래도 저장하시겠습니까? [그래도 저장/수정/취소]: ").strip()


def _render_diff_item(payload: dict, index: int, total: int) -> bool:
    _print(f"[{index}/{total}] {payload['action'].upper()} {payload['category']}: {payload['entity_id']}")
    _print(f"  근거: {payload['reason']}")
    _print(f"  필드: {payload['fields']}")
    answer = _prompt("  승인하시겠습니까? (y/n): ").strip().lower()
    return answer == "y"


_DECISION_RENDERERS = {
    "entity_candidates": _render_entity_candidates,
    "entity_name": _render_entity_name,
    "entity_terminal_status": _render_entity_terminal_status,
    "entity_required_field": _render_entity_required_field,
    "hard_check_warning": _render_hard_check_warning,
    "rag_judgment": _render_rag_judgment,
}


def _print_result(result: dict) -> None:
    status = result.get("status")
    if status == "error":
        _print(f"입력 오류: {result['message']}")
    elif status == "rejected" and result.get("stage") == "hard_check":
        blocking = [c for c in result.get("conflicts", []) if c.severity == "blocking"]
        for c in blocking:
            _print(f"  - [{c.check_type}] {c.entity_id}: {c.reason}")
        _print("하드체크 결과에 따라 저장이 중단되었습니다.")
    elif status == "rejected" and result.get("stage") == "rag_check":
        _print("RAG 검증 결과에 따라 저장이 중단되었습니다.")
    elif status == "no_changes":
        _print("승인된 변경사항이 없어 저장할 내용이 없습니다.")
    elif status == "saved":
        applied = result.get("applied", [])
        _print("저장 완료: " + ", ".join(_describe_applied(c) for c in applied))


def run_pipeline_interactive(user_input: str) -> dict:
    """Drive a single input through pipeline_session's state machine,
    rendering every pending decision as the same CLI prompt run_pipeline used
    to show directly. Kept separate from cli_loop's while-loop so a single
    input can be exercised (e.g. from tests) without the surrounding REPL."""
    session = pipeline_session.start_session(user_input)
    diff_total = None
    diff_index = 0

    while session.pending_decision is not None:
        decision = session.pending_decision
        if decision.decision_type == "diff_item":
            if diff_total is None:
                diff_total = len(session.diff)
            diff_index += 1
            response = _render_diff_item(decision.payload, diff_index, diff_total)
        else:
            response = _DECISION_RENDERERS[decision.decision_type](decision.payload)
        session = pipeline_session.resume_session(session.session_id, response)

    _print_result(session.result)
    return session.result


def cli_loop() -> None:
    _print("Lore Builder CLI. 종료하려면 '종료'를 입력하세요.")
    while True:
        try:
            user_input = _prompt("\n입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\n종료합니다.")
            break
        if user_input == "종료":
            _print("종료합니다.")
            break
        if not user_input:
            continue
        try:
            run_pipeline_interactive(user_input)
        except Exception as exc:
            _print(f"처리 중 오류가 발생했습니다: {exc}")


if __name__ == "__main__":
    cli_loop()
