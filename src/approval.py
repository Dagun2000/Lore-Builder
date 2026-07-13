"""Interactive approval loop — Phase 5.

All CLI I/O funnels through _prompt/_print, the same seam mapping.py uses,
so a future GUI can swap these out without touching the review logic itself.
"""

from . import storage


def _prompt(message: str) -> str:
    return input(message)


def _print(message: str = "") -> None:
    print(message)


def review_hard_check_conflicts(conflicts: list) -> bool:
    blocking = [c for c in conflicts if c.severity == "blocking"]
    if blocking:
        _print("다음 항목이 하드체크를 위반해 저장이 거부되었습니다:")
        for c in blocking:
            _print(f"  - [{c.check_type}] {c.entity_id}: {c.reason}")
        return False

    for c in [c for c in conflicts if c.severity == "warning"]:
        _print(f"[경고] {c.entity_id}: {c.reason}")
        answer = _prompt("그래도 저장하시겠습니까? [그래도 저장/수정/취소]: ").strip()

        if answer == "그래도 저장":
            if c.check_type == "lifespan":
                # Persist the ack now so the same warning doesn't fire again
                # on this entity next time (Phase 1's check_lifespan_violation
                # short-circuits once lifespan_check_ack is true).
                storage.save_entity(
                    "character", c.entity_id, {"lifespan_check_ack": True}
                )
            continue
        if answer == "수정":
            _print("입력을 다시 확인해주세요.")
            return False
        # "취소" or anything unrecognized: stop.
        _print("저장이 취소되었습니다.")
        return False

    return True


def review_rag_judgments(judgments: list) -> bool:
    for j in judgments:
        if j.type == "clears_status":
            _print(f"[안내] '{j.status_effect_id}' 상태가 해제됩니다. ({j.reason})")
            continue

        _print(f"[{j.type}] {j.reason}")
        answer = _prompt("그래도 저장하시겠습니까? [그래도 저장/수정/취소]: ").strip()

        if answer == "그래도 저장":
            continue
        if answer == "수정":
            _print("입력을 다시 확인해주세요.")
            return False
        _print("저장이 취소되었습니다.")
        return False

    return True


def review_diff(diff: list) -> list:
    approved = []
    total = len(diff)

    for i, item in enumerate(diff, start=1):
        _print(f"[{i}/{total}] {item.action.upper()} {item.category}: {item.entity_id}")
        _print(f"  근거: {item.reason}")
        _print(f"  필드: {item.fields}")
        answer = _prompt("  승인하시겠습니까? (y/n): ").strip().lower()
        if answer == "y":
            approved.append(item)

    return approved
