"""Existing-entity field update — Phase 6, Track B rewritten in Phase 10.

Deliberately separate from Step 4's rag_check.py: everything here is pure
retrieval, never an LLM contradiction judgment. Whether a related record
actually conflicts with the new value is left entirely to the human
reviewing update_field_flow's output — see the module docstring in
rag_check.py if this distinction matters for a future change.

Two independent tracks, both triggered by update_field_flow, never just one:
  - Track A (update_structured_field): re-runs Phase 1's hard_check on any
    field Phase 1 already reasons about, reusing Phase 5's approval loop.
  - Track B (find_related_context): every event entity_id.event_ids points
    at — Phase 10 retired the relationship table + Chroma-similarity search
    this used to run; event_ids is now the complete, exact list of what's
    related, so there's nothing left to rank by similarity for.
"""

from dataclasses import dataclass

from . import approval, flags, hard_check, schema, storage

_NAME_BEARING_CATEGORIES = ("character", "location", "faction", "artifact", "race")

# Fields Phase 1's hard checks read even though the field itself carries no
# `role` tag — e.g. check_lifespan_violation reads character.race. Extend
# this if Phase 1 grows more such cross-field dependencies.
_STRUCTURED_FIELD_OVERRIDES = {
    "character": {"race"},
}


@dataclass
class RelatedDoc:
    entity_id: str  # always a timeline (event_) id — Phase 10
    source: str  # "point" | "duration"
    relation: str  # duration's predicate, or "event" for a point record
    text: str
    relevance_rank: int  # chronological position (1 = earliest); not a similarity score


def _prompt(message: str) -> str:
    return input(message)


def _print(message: str = "") -> None:
    print(message)


def is_structured_field(category: str, field_name: str) -> bool:
    field_defs = {f["name"]: f for f in schema.get_fields(category)}
    field_def = field_defs.get(field_name)
    if field_def and field_def.get("role"):
        return True
    return field_name in _STRUCTURED_FIELD_OVERRIDES.get(category, set())


# ---------------------------------------------------------------------------
# Track A — reuse Phase 1's hard checks
# ---------------------------------------------------------------------------

def update_structured_field(entity_id: str, field_name: str, new_value):
    """Write `new_value`, re-run Phase 1's hard checks, and route the result
    through Phase 5's approval loop. If rejected (blocking, or a warning the
    user declined), the field is rolled back to its previous value — nothing
    Phase 1 would flag is left committed. Returns (approved: bool, conflicts:
    list[Conflict])."""
    category = schema.category_from_id(entity_id)
    if category is None:
        raise ValueError(f"알 수 없는 entity_id입니다: {entity_id}")

    previous = storage.get_entity(category, entity_id) or {}
    previous_value = previous.get(field_name)

    storage.save_entity(category, entity_id, {field_name: new_value})
    conflicts = hard_check.run_hard_checks(category, entity_id)
    approved = approval.review_hard_check_conflicts(conflicts)

    if not approved:
        storage.save_entity(category, entity_id, {field_name: previous_value})

    return approved, conflicts


# ---------------------------------------------------------------------------
# Track B — full recall via event_ids, no ranking, no LLM judgment
# ---------------------------------------------------------------------------

def find_related_context(entity_id: str) -> list:
    """Every event entity_id.event_ids points at, chronological — the
    complete, exact list of what's related to this entity. Phase 10 retired
    the old relationship-table-plus-Chroma-similarity-search version of this
    function: event_ids already IS full recall, so there's nothing left to
    search or rank by similarity for."""
    records = storage.get_events_for_entity(entity_id)
    docs = []
    for rank, record in enumerate(records, start=1):
        if record.get("year") is not None:
            source, relation, year_label = "point", "event", str(record["year"])
        else:
            source = "duration"
            relation = record.get("predicate") or ""
            end = record.get("end_year")
            year_label = f"{record.get('start_year')}~{end if end is not None else '현재'}"
        docs.append(
            RelatedDoc(
                entity_id=record["id"],
                source=source,
                relation=relation,
                text=f"[{year_label}] {record.get('notes') or ''}".strip(),
                relevance_rank=rank,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# CLI-facing flow (GUI can call this directly instead of detail_panel.py)
# ---------------------------------------------------------------------------

def _print_related_context(related_docs: list, show_top: int = 3) -> None:
    if not related_docs:
        _print("관련 기록이 없습니다.")
        return

    _print(f"관련 기록 (관련도 순, 총 {len(related_docs)}건):")
    for doc in related_docs[:show_top]:
        _print(f"  {doc.relevance_rank}. {doc.entity_id} ({doc.source}: {doc.relation})")
        _print(f'     "{doc.text}"')

    remaining = related_docs[show_top:]
    if not remaining:
        return

    _print(f"[더 보기 ({len(remaining)}건 남음)]")
    answer = _prompt("더 보기를 원하시면 'more', 넘어가려면 Enter: ").strip().lower()
    if answer == "more":
        for doc in remaining:
            _print(f"  {doc.relevance_rank}. {doc.entity_id} ({doc.source}: {doc.relation})")
            _print(f'     "{doc.text}"')


def _flag_related_docs(related_docs: list, flagged_from: str) -> list:
    """Ask which (if any) of the just-shown related records need a closer
    look later. Completely independent of the field save itself — flagging
    something here neither blocks nor auto-fixes anything; it's just added
    to the review list (see flags.py). Returns the Flag objects created."""
    if not related_docs:
        return []

    answer = _prompt(
        "수정이 필요해 보이는 항목이 있나요? 번호를 입력하세요 (쉼표로 여러 개, 없으면 Enter): "
    ).strip()
    if not answer:
        return []

    by_rank = {doc.relevance_rank: doc for doc in related_docs}
    created = []
    for raw_index in answer.split(","):
        raw_index = raw_index.strip()
        if not raw_index.isdigit():
            continue
        doc = by_rank.get(int(raw_index))
        if doc is None:
            continue
        reason = _prompt(
            f"{doc.entity_id}에 플래그를 걸었습니다. (사유를 남기시겠습니까? 없으면 Enter): "
        ).strip()
        created.append(flags.add_flag(doc.entity_id, flagged_from, reason or None))

    return created


def update_field_flow(entity_id: str, field_name: str, new_value) -> dict:
    category = schema.category_from_id(entity_id)
    if category is None:
        raise ValueError(f"알 수 없는 entity_id입니다: {entity_id}")

    structured = is_structured_field(category, field_name)
    conflicts = []
    previous_value = (storage.get_entity(category, entity_id) or {}).get(field_name)

    if structured:
        approved, conflicts = update_structured_field(entity_id, field_name, new_value)
        if not approved:
            # update_structured_field already rolled itself back.
            return {
                "status": "rejected",
                "entity_id": entity_id,
                "field_name": field_name,
                "conflicts": conflicts,
                "related_docs": [],
                "flagged": [],
            }

    related_docs = find_related_context(entity_id)

    _print(f'[{entity_id}]의 {field_name}을(를) "{new_value}"(으)로 저장하려 합니다.')
    _print_related_context(related_docs)

    # Flagging is independent of the save decision below — it's asked here
    # (right after the related records are shown) but neither blocks nor
    # influences whether the field update itself proceeds.
    flagged = _flag_related_docs(
        related_docs, flagged_from=f"{entity_id}의 {field_name} 수정 중 발견"
    )

    answer = _prompt("확인 후 그대로 저장하시겠습니까? (y/n): ").strip().lower()
    if answer != "y":
        if structured:
            # Track A already committed new_value (its own check passed);
            # undo it now that the human vetoed it at the related-context
            # review — restore exactly what was there before this call.
            storage.save_entity(category, entity_id, {field_name: previous_value})
        _print("저장이 취소되었습니다. 값을 다시 입력해주세요.")
        return {
            "status": "cancelled",
            "entity_id": entity_id,
            "field_name": field_name,
            "conflicts": conflicts,
            "related_docs": related_docs,
            "flagged": flagged,
        }

    if not structured:
        storage.save_entity(category, entity_id, {field_name: new_value})

    _print(f"{entity_id}.{field_name} 값이 저장되었습니다.")

    # This entity just got fixed, so whatever flags were raised against it
    # (from any edit context, not just this one) no longer apply — clearing
    # them here is what makes list_flags_deduped()'s one-line-per-entity view
    # actually stay accurate instead of showing a "fixed" entity forever.
    cleared = flags.clear_flags_for_entity(entity_id)
    if cleared:
        _print(f"이 엔티티에 걸려있던 플래그 {cleared}건이 자동 해제됐습니다.")

    return {
        "status": "saved",
        "entity_id": entity_id,
        "field_name": field_name,
        "conflicts": conflicts,
        "related_docs": related_docs,
        "flagged": flagged,
        "cleared_flags": cleared,
    }
