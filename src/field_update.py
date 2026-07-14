"""Existing-entity field update — Phase 6.

Deliberately separate from Phase 3's rag_check.py: everything here is pure
retrieval + ranking, never an LLM contradiction judgment. Whether a related
record actually conflicts with the new value is left entirely to the human
reviewing update_field_flow's output — see the module docstring in
rag_check.py if this distinction matters for a future change.

Two independent tracks, both triggered by update_field_flow, never just one:
  - Track A (update_structured_field): re-runs Phase 1's hard_check on any
    field Phase 1 already reasons about, reusing Phase 5's approval loop.
  - Track B (find_related_context): full-recall search over every
    relationship/timeline record touching the entity, ranked (not filtered)
    by similarity to the new value via Chroma — always runs, regardless of
    field type.
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
    entity_id: str
    source: str  # "relationship" | "timeline"
    relation: str
    text: str
    relevance_rank: int


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
# Track B — full-recall search + ranking, no LLM judgment
# ---------------------------------------------------------------------------

def _gather_relationship_docs(entity_id: str) -> list:
    """Relationships whose counterpart is a real entity (not a timeline
    event — those are handled by _gather_timeline_docs so a relationship
    row pointing at an event_ isn't counted in both buckets)."""
    conn = storage.get_connection()
    storage.init_db(conn)

    docs = []
    for row in conn.execute(
        'SELECT subject, predicate, object, notes FROM "relationship" '
        'WHERE subject = ? OR object = ?',
        (entity_id, entity_id),
    ):
        other = row["object"] if row["subject"] == entity_id else row["subject"]
        if not other:
            continue

        other_category = schema.category_from_id(other)
        if other_category is None or other_category == "timeline":
            continue

        other_entity = storage.get_entity(other_category, other)
        counterpart_notes = (other_entity or {}).get("notes") or ""
        relationship_notes = row["notes"] or ""
        text = " ".join(t for t in (counterpart_notes, relationship_notes) if t)
        if not text:
            continue

        docs.append(
            RelatedDoc(
                entity_id=other,
                source="relationship",
                relation=row["predicate"],
                text=text,
                relevance_rank=-1,
            )
        )

    conn.close()
    return docs


def _gather_timeline_docs(entity_id: str) -> list:
    docs = []
    for event_id in storage.find_related_timeline_ids(entity_id):
        event = storage.get_entity("timeline", event_id)
        text = (event or {}).get("notes") or ""
        if not text:
            continue
        docs.append(
            RelatedDoc(
                entity_id=event_id,
                source="timeline",
                relation="event",
                text=text,
                relevance_rank=-1,
            )
        )
    return docs


def _resolve_display_value(value) -> str:
    """If `value` is itself an entity id (updating a reference-type field,
    e.g. race/faction to another entity), use that entity's human-readable
    `name` for the ranking query instead of the raw id. Verified this
    matters: querying Chroma with the raw id "faction_mercenary_guild"
    ranked the correct doc *last* of 4 candidates; querying with its name
    "용병 길드" ranked it first — Chroma's default embedding model doesn't
    treat snake_case/prefixed ids as meaningful tokens against Korean text."""
    if not isinstance(value, str):
        return str(value)
    category = schema.category_from_id(value)
    if category is None:
        return value
    entity = storage.get_entity(category, value)
    if entity and entity.get("name"):
        return entity["name"]
    return value


def find_related_context(entity_id: str, updated_field: str, new_value) -> list:
    """Every relationship/timeline record touching entity_id, ranked by
    similarity to the new value — full population, nothing dropped for
    being low-ranked. No LLM call happens here.

    The query is the (name-resolved) value suffixed with a generic Korean
    connector ("{value} 관련" — "related to {value}"), not the raw English
    `updated_field` name. Measured both matter: prefixing the raw field name
    (e.g. "faction: 용병 길드") pushed the correct doc out of first place.
    The bare value alone is also unreliable once two candidate docs share a
    keyword unrelated to the actual question — e.g. querying "철왕국" alone
    let a doc about char_mira living in the same city outrank the actual
    faction-membership doc, because both mention "은빛도시". Appending "관련"
    consistently fixed that without needing per-field Korean vocabulary."""
    docs = _gather_relationship_docs(entity_id) + _gather_timeline_docs(entity_id)
    if not docs:
        return []

    doc_ids = [d.entity_id for d in docs]
    query_text = f"{_resolve_display_value(new_value)} 관련"
    results = storage.query_chroma(query_text, top_k=len(doc_ids), ids=doc_ids)
    ranked_ids = (results.get("ids") or [[]])[0]
    order = {doc_id: i for i, doc_id in enumerate(ranked_ids)}
    tail_rank = len(ranked_ids)

    docs.sort(key=lambda d: order.get(d.entity_id, tail_rank))
    for rank, doc in enumerate(docs, start=1):
        doc.relevance_rank = rank

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

    related_docs = find_related_context(entity_id, field_name, new_value)

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
