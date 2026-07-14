"""Detail-panel CLI — Phase 6's standalone entry point.

Deliberately separate from main.py: main.run_pipeline is for new events and
relationships only ("what happened"). This is for editing a field on an
entity that already exists ("look up X, change Y") — a different mental
model with a different trigger (picking an entity, not describing an
event). A future GUI's detail panel will call field_update.update_field_flow
directly and skip this CLI shell entirely, the same way it would skip
main.cli_loop() and call main.run_pipeline directly.
"""

import sys

# See main.py's identical block for why: Windows consoles default stdio to
# the active code page (cp949 on Korean Windows), not UTF-8.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

if __package__:
    from . import field_update, flags, mapping, schema, storage
else:  # allows `python src/detail_panel.py` to run directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import field_update, flags, mapping, schema, storage

_NAME_BEARING_CATEGORIES = ("character", "location", "faction", "artifact", "race")


def _prompt(message: str) -> str:
    return input(message)


def _print(message: str = "") -> None:
    print(message)


def _select_candidate(query: str, candidates: list, allow_cancel: bool = False) -> str | None:
    """Numbered pick list. `allow_cancel` adds a "0. 취소" option — used for
    partial/substring matches, which must never auto-connect even when
    there's only one candidate, unlike a unique exact match."""
    if len(candidates) > 1:
        _print(f"'{query}'와(과) 일치하는 후보가 여러 개입니다:")
    else:
        _print(f"'{query}'와(과) 정확히 일치하진 않지만 비슷한 후보가 있습니다:")
    for i, candidate_id in enumerate(candidates, start=1):
        _print(f"  {i}. {candidate_id}")
    if allow_cancel:
        _print("  0. 취소 (다시 검색)")

    low = 0 if allow_cancel else 1
    choice = _prompt(f"번호를 선택하세요 ({low}-{len(candidates)}): ").strip()
    if choice.isdigit():
        idx = int(choice)
        if allow_cancel and idx == 0:
            return None
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
    _print("잘못된 번호입니다.")
    return None


def _find_entity_interactive(query: str) -> str | None:
    category = schema.category_from_id(query)
    if category and storage.entity_exists(category, query):
        return query

    exact_candidates = []
    partial_candidates = []
    for candidate_category in _NAME_BEARING_CATEGORIES:
        exact, partial = mapping.find_existing_matches(query, candidate_category)
        exact_candidates.extend(exact)
        partial_candidates.extend(partial)

    if len(exact_candidates) == 1:
        return exact_candidates[0]
    if len(exact_candidates) > 1:
        return _select_candidate(query, exact_candidates)

    if partial_candidates:
        # No exact match — even a single substring hit must be confirmed,
        # never silently auto-connected (e.g. "주점" -> "검은 산양 주점").
        return _select_candidate(query, partial_candidates, allow_cancel=True)

    _print(f"'{query}'와(과) 일치하는 엔티티를 찾지 못했습니다.")
    return None


def _pick_field_interactive(category: str) -> str | None:
    field_defs = schema.get_fields(category)
    _print(f"[{category}] 필드 목록:")
    for i, f in enumerate(field_defs, start=1):
        marker = "*" if field_update.is_structured_field(category, f["name"]) else " "
        _print(f"  {i}.{marker} {f['name']}")
    choice = _prompt("수정할 필드 번호를 입력하세요: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(field_defs):
        return field_defs[int(choice) - 1]["name"]
    _print("잘못된 번호입니다.")
    return None


def _print_flags_list() -> None:
    entries = flags.list_flags_deduped()
    if not entries:
        _print("플래그된 항목이 없습니다.")
        return

    _print(f"현재 플래그된 항목 ({len(entries)}건):")
    for flag in entries:
        reason_text = f'"{flag.reason}"' if flag.reason else "(사유 없음)"
        _print(f"[{flag.id}] {flag.entity_id} — {reason_text}")
        _print(f"    ({flag.flagged_from}, {flag.created_at.split('T')[0]})")


def detail_panel_loop() -> None:
    _print("Lore Builder — 디테일 패널. 종료하려면 '종료'를 입력하세요.")
    while True:
        try:
            query = _prompt("\n수정할 엔티티 (id 또는 이름, '목록'으로 플래그 확인)> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\n종료합니다.")
            break

        if query == "종료":
            _print("종료합니다.")
            break
        if query in ("목록", "flags"):
            _print_flags_list()
            continue
        if not query:
            continue

        entity_id = _find_entity_interactive(query)
        if entity_id is None:
            continue

        category = schema.category_from_id(entity_id)
        field_name = _pick_field_interactive(category)
        if field_name is None:
            continue

        field_def = next(f for f in schema.get_fields(category) if f["name"] == field_name)
        raw_value = _prompt(f"{field_name} 새 값 입력: ").strip()
        new_value = schema.coerce_value(field_def, raw_value)

        try:
            field_update.update_field_flow(entity_id, field_name, new_value)
        except Exception as exc:
            _print(f"처리 중 오류가 발생했습니다: {exc}")


if __name__ == "__main__":
    detail_panel_loop()
