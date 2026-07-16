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

Phase 10 patch 8 adds a third track, find_relevant_context, used by the GUI's
field-edit screen instead of Track B: rather than dumping every event this
entity is pointed at (noisy — a destroyed_year edit doesn't need to see the
item's forging event), it walks one hop out to whichever *other* entities
share an event with this one, and asks an LLM in a single batched call
whether each candidate's own notes/fields look relevant to *this specific
edit*. Like Track B this only surfaces candidates for a human to look at —
it is not a save-time contradiction verdict (that's still rag_check.
check_notes_conflict, a separate call at a separate point in the pipeline).
"""

import json
import re
from dataclasses import dataclass

from . import approval, config, flags, hard_check, rag_check, schema, storage

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
# Track C (Phase 10 patch 8) — 1-hop, edit-aware relevance search
# ---------------------------------------------------------------------------

_AGE_FOCUS_DESCRIPTION = (
    "나이, 연령, 나이 제한(예: '몇 살 이상만'), 세월의 경과, 늙음/젊음 등 나이·시간과 "
    "관련된 서술"
)


@dataclass
class RelevantMatch:
    entity_id: str  # a 1-hop entity, or (name edits only) a timeline event id
    reason: str  # the candidate text that triggered the match


def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.get_model("reasoning"), temperature=0)


def _invoke_llm(prompt: str) -> str:
    response = _get_llm().invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾을 수 없습니다: {raw!r}")
    return json.loads(match.group(0))


def _one_hop_entities(entity_id: str) -> set:
    """Every other entity sharing an event with entity_id, deduplicated to
    unique entities rather than counted per event (Phase 10 patch 8, 2-1) —
    a character entangled with the same faction across a hundred events
    still contributes that faction exactly once. Point events don't store
    their own participant list, so who else is connected is only knowable
    via the reverse lookup (storage.find_entities_referencing_event);
    duration events already carry entity/target directly."""
    connected = set()
    for record in storage.get_events_for_entity(entity_id):
        if record.get("year") is not None:  # point event
            participants = [
                eid for _category, eid in storage.find_entities_referencing_event(record["id"])
            ]
        else:  # duration event
            participants = [record.get("entity"), record.get("target")]
        for participant in participants:
            if participant and participant != entity_id:
                connected.add(participant)
    return connected


def _is_age_field(category: str, field_name: str) -> bool:
    """Only character birth_year/death_year carry an "age" meaning — a
    faction's founded_year or an artifact's created_year don't feed any
    lifespan-style hard check, so they're treated as ordinary structured
    fields instead (Phase 10 patch 8, 2-2)."""
    if category != "character":
        return False
    field_def = next((f for f in schema.get_fields(category) if f["name"] == field_name), None)
    return bool(field_def and field_def.get("role") in ("lifecycle_start", "lifecycle_end"))


def _notes_diff(old_text, new_text) -> str:
    """The newly added/changed portion of a notes edit, not the whole
    field (Phase 10 patch 8, 2-3) — the common case (appending a sentence)
    is handled exactly; a full rewrite falls back to the whole new text
    since there's no clean "added part" to isolate."""
    old_text = old_text or ""
    new_text = new_text or ""
    if old_text and old_text in new_text:
        remainder = new_text.replace(old_text, "", 1).strip()
        return remainder or new_text
    return new_text


def _candidate_text(entity_id: str, include_fields: bool) -> str:
    category = schema.category_from_id(entity_id)
    if category is None:
        return ""
    record = storage.get_entity(category, entity_id) or {}
    parts = []
    if include_fields:
        summary = rag_check.entity_field_summary(record)
        if summary:
            parts.append(summary)
    if record.get("notes"):
        parts.append(record["notes"])
    return " / ".join(parts)


def _judge_relevance(candidates: list, focus: str) -> set:
    """One batched LLM call comparing every candidate against `focus` at
    once (Phase 10 patch 8, 2-4) — never per-candidate, never a vector/
    embedding search, never a hardcoded synonym table. The same style of
    direct LLM comparison rag_check.check_notes_conflict already uses
    (which is why synonyms/rephrasings like "여성/여자/계집" work without
    ever being enumerated anywhere). Purely a "worth a human look" filter —
    not a save-time contradiction verdict."""
    candidates = [(eid, text) for eid, text in candidates if text]
    if not candidates:
        return set()

    block = "\n".join(f"- {eid}: {text}" for eid, text in candidates)
    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 보조 검색기다. 아래 후보 목록 중, 다음 "
        f"기준과 관련이 있어 보이는 항목을 전부 골라라.\n\n기준: {focus}\n\n"
        f"후보 목록:\n{block}\n\n"
        "동의어나 표현이 달라도(예: '여성'과 '여자') 의미가 통하면 관련 있다고 판단하라. "
        "확신이 없어도 조금이라도 관련 가능성이 있으면 포함시켜라 — 이건 최종 판단이 아니라 "
        "사람이 검토할 후보를 추리는 것뿐이다.\n\n"
        '아래 JSON 형식으로만 답하라: {"relevant_entity_ids": ["entity_id", ...]}\n'
    )
    print(f"[field_update] 관련성 판단 후보 {len(candidates)}건, 기준: {focus}")
    try:
        data = _extract_json(_invoke_llm(prompt))
    except Exception as exc:
        print(f"[field_update] 관련성 판단 실패, 빈 결과로 처리: {exc}")
        return set()
    relevant_ids = set(data.get("relevant_entity_ids") or [])
    print(f"[field_update] 관련성 판단 결과: {sorted(relevant_ids) or '(없음)'}")
    return relevant_ids


def find_relevant_context(entity_id: str, field_name: str, new_value) -> list:
    """Edit-aware related-context search (Phase 10 patch 8) — what's shown
    to the GUI's field-edit screen instead of the old full event listing.
    Search scope is asymmetric depending on what's being edited:
      - name: not a relevance judgment at all — the old name is searched
        verbatim against this entity's own past event notes, since a
        rename is a "find the frozen-in-prose old string" problem, not a
        "does this fact still hold" problem.
      - notes: only the newly added/changed portion is searched, against
        every 1-hop entity's fields *and* notes (prose can contradict any
        settled fact).
      - an age field (character birth/death year): the existing hard check
        is untouched; this *additionally* searches 1-hop notes for an
        age/time-related mention, since "must be over 100 years old" is a
        prose constraint no deterministic check can catch.
      - anything else structured (gender, category, domain, ...): 1-hop
        notes only, searched with both the old and new value — a rule
        like "남성만 있는 형제단" would otherwise be missed when flipping
        gender away from the value the rule was written against.
    """
    category = schema.category_from_id(entity_id)
    if category is None:
        return []

    current = storage.get_entity(category, entity_id) or {}
    previous_value = current.get(field_name)

    if field_name == "name":
        matches = []
        for record in storage.get_events_for_entity(entity_id):
            notes = record.get("notes") or ""
            if previous_value and previous_value in notes:
                matches.append(RelevantMatch(entity_id=record["id"], reason=notes))
        return matches

    one_hop = _one_hop_entities(entity_id)

    if field_name == "notes":
        diff = _notes_diff(previous_value, new_value)
        focus = f'"{diff}"라는 서술과 관련되거나 모순될 수 있는 내용'
        candidates = [(eid, _candidate_text(eid, include_fields=True)) for eid in one_hop]
    elif _is_age_field(category, field_name):
        focus = _AGE_FOCUS_DESCRIPTION
        candidates = [(eid, _candidate_text(eid, include_fields=False)) for eid in one_hop]
    else:
        focus = f'"{previous_value}" 또는 "{new_value}"와(과) 관련되거나 전제로 하는 규칙/서술'
        candidates = [(eid, _candidate_text(eid, include_fields=False)) for eid in one_hop]

    relevant_ids = _judge_relevance(candidates, focus)
    return [
        RelevantMatch(entity_id=eid, reason=text)
        for eid, text in candidates
        if eid in relevant_ids
    ]


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
