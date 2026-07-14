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


def review_confirmation_needed(confirmation) -> bool:
    """archivist.build_diff couldn't produce a coherent record at all —
    either the sentence was judged to describe more than one event/status,
    or a "clear" named a status/relationship that isn't actually open.
    There's no partial diff to fall back to either way, so the only choices
    are acknowledge-and-stop or cancel — both end up saving nothing."""
    _print(f"[확인 필요] {confirmation.reason}")
    answer = _prompt("계속 진행하시겠습니까? [계속 진행/취소]: ").strip()
    return answer == "계속 진행"


def review_diff(diff: list) -> list:
    """One bundled decision for the whole diff (Phase 10 patch), not one per
    ChangeItem — a diff is always "one primary timeline record + pointer/
    cache updates that belong to it", so there's nothing meaningful to
    approve item-by-item. Shows the primary record plus who else gets
    touched as information; 저장 applies everything, 취소 applies nothing."""
    if not diff:
        return []

    primary = next((c for c in diff if c.category == "timeline"), diff[0])
    affected = [c.entity_id for c in diff if c is not primary]

    _print(f"{primary.action.upper()} {primary.category}: {primary.entity_id}")
    _print(f"  근거: {primary.reason}")
    _print(f"  필드: {primary.fields}")
    if affected:
        _print(f"  함께 갱신되는 엔티티: {', '.join(affected)}")

    answer = _prompt("저장하시겠습니까? [저장/취소]: ").strip()
    return diff if answer == "저장" else []
