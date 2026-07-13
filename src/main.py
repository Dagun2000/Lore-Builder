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
            run_pipeline(user_input)
        except Exception as exc:
            _print(f"처리 중 오류가 발생했습니다: {exc}")


if __name__ == "__main__":
    cli_loop()
