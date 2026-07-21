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
entity is pointed at unfiltered, it asks an LLM in a single batched call
whether each of entity_id's own events looks relevant to *this specific
edit*. Like Track B this only surfaces candidates for a human to look at —
it is not a save-time contradiction verdict (that's still rag_check.
check_notes_conflict, a separate call at a separate point in the pipeline).

An earlier version of this search walked one hop out to whichever *other*
entities shared an event with this one (a faction, a location, ...) instead
of the events themselves — found lacking in practice: editing a character's
own notes to drop a sentence about a past brawl surfaced the tavern it
happened at as "related," which isn't wrong or stale in any way and isn't
something there's anything to actually do about. The event describing
exactly what's being removed is the only thing that's both genuinely
inconsistent with the edit and actually actionable (edit or delete it), so
that's what the candidate pool is now built from.
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
    return config.get_chat_model("reasoning", temperature=0)


def _invoke_llm(prompt: str) -> str:
    response = _get_llm().invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾을 수 없습니다: {raw!r}")
    return json.loads(match.group(0))


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
    """The specific part that changed, not the whole field (Phase 10 patch
    8, 2-3) — handles the two common edit shapes precisely: appending a
    sentence (old_text is a prefix/substring of new_text — return the
    newly added remainder) and *removing* one (new_text is a substring of
    old_text — return the removed portion itself). The removal case was
    missing entirely at first: dropping a sentence from notes fell through
    to the "return new_text" fallback, meaning the relevance search's
    focus became whatever prose was left over, never the fact that was
    actually just taken out — so the one thing genuinely inconsistent with
    the edit (an event describing exactly what got removed) could never be
    surfaced, no matter how relevant it actually was. A full rewrite
    (neither contains the other) still falls back to the whole new text —
    there's no clean single "changed part" to isolate there."""
    old_text = old_text or ""
    new_text = new_text or ""
    if old_text and old_text in new_text:
        remainder = new_text.replace(old_text, "", 1).strip()
        return remainder or new_text
    if new_text in old_text:
        # No truthy guard on new_text here (unlike the append branch above)
        # — new_text == "" (notes cleared out entirely) is a real case and
        # "" is trivially "in" anything, which is exactly the behavior
        # wanted: the whole old_text comes back as what was removed.
        removed = old_text.replace(new_text, "", 1).strip()
        return removed or old_text
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


def _judge_relevance(candidates: list, focus: str) -> list:
    """One batched LLM call comparing every candidate against `focus` at
    once (Phase 10 patch 8, 2-4) — never per-candidate, never a vector/
    embedding search, never a hardcoded synonym table. The same style of
    direct LLM comparison rag_check.check_notes_conflict already uses
    (which is why synonyms/rephrasings like "여성/여자/계집" work without
    ever being enumerated anywhere). Purely a "worth a human look" filter —
    not a save-time contradiction verdict.

    Returns entity_ids in the order the model listed them (asked to rank
    most-relevant-first), deduplicated and filtered to real candidates —
    not a set. A set silently discarded whatever order the model's own
    response carried, so even when the genuinely correct match was found,
    it could land anywhere in the final displayed list instead of first
    (caught in practice: the actually-correct match ranked 3rd of 4).

    Tightened to favor precision over recall (an earlier version
    explicitly told the model to include a candidate "even without
    confidence, if there's even a little chance of relevance" — reasoning
    that this is just a pre-filter for a human to review, not a final
    verdict. In practice that produced noisy over-inclusion: candidates
    with no real bearing on the edit still showed up alongside the one
    genuine match. If the one clearly-relevant record has already been
    deleted and nothing else is actually related, the correct result is
    an empty list, not a forced best-effort guess.)"""
    candidates = [(eid, text) for eid, text in candidates if text]
    if not candidates:
        return []
    candidate_ids = {eid for eid, _ in candidates}

    block = "\n".join(f"- {eid}: {text}" for eid, text in candidates)
    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 보조 검색기다. 아래 후보 목록 중, 다음 "
        f"기준과 실제로 관련이 있는 항목만 골라라.\n\n기준: {focus}\n\n"
        f"후보 목록:\n{block}\n\n"
        "동의어나 표현이 달라도(예: '여성'과 '여자') 의미가 통하면 관련 있다고 판단하라. "
        "하지만 명확한 연관이 없는 후보를 억지로 포함시키지 마라 — 조금이라도 관련 "
        "가능성이 있다는 이유만으로 포함시키지 말고, 실제로 그 서술과 관련되거나 그 "
        "서술로 인해 다시 검토가 필요한 항목만 골라라. 진짜로 관련된 항목이 하나도 없다면 "
        "빈 배열을 반환하는 것이 맞는 답이다.\n\n"
        "관련 있다고 고른 항목은 관련성이 높은 순서대로 나열하라.\n\n"
        '아래 JSON 형식으로만 답하라: {"relevant_entity_ids": ["entity_id", ...]}\n'
    )
    print(f"[field_update] 관련성 판단 후보 {len(candidates)}건, 기준: {focus}")
    try:
        data = _extract_json(_invoke_llm(prompt))
    except Exception as exc:
        print(f"[field_update] 관련성 판단 실패, 빈 결과로 처리: {exc}")
        return []
    raw_ids = data.get("relevant_entity_ids") or []
    seen = set()
    ordered_ids = []
    for eid in raw_ids:
        if eid in candidate_ids and eid not in seen:
            seen.add(eid)
            ordered_ids.append(eid)
    print(f"[field_update] 관련성 판단 결과: {ordered_ids or '(없음)'}")
    return ordered_ids


def find_relevant_context(entity_id: str, field_name: str, new_value) -> list:
    """Edit-aware related-context search (Phase 10 patch 8) — what's shown
    to the GUI's field-edit screen instead of the old full event listing.
    Search scope is asymmetric depending on what's being edited:
      - name: not a relevance judgment at all — the old name is searched
        verbatim against this entity's own past event notes, since a
        rename is a "find the frozen-in-prose old string" problem, not a
        "does this fact still hold" problem.
      - notes: only the newly added/changed portion is searched, against
        entity_id's own events (prose can contradict any of them).
      - an age field (character birth/death year): the existing hard check
        is untouched; this *additionally* searches entity_id's own events
        for an age/time-related mention, since "must be over 100 years
        old" is a prose constraint no deterministic check can catch.
      - anything else structured (gender, category, domain, ...):
        entity_id's own events, searched with both the old and new value —
        a rule like "남성만 있는 형제단" would otherwise be missed when
        flipping gender away from the value the rule was written against.

    Candidates are always entity_id's own timeline events, never the other
    entities that happen to share them — see the module docstring for why.
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

    # Candidates are entity_id's own timeline events, not the other
    # entities that happen to share them (Phase 10 patch 8 originally
    # searched one-hop *entities* here; found lacking in practice — editing
    # 데이비드's own notes to drop "검은 염소 주점에서 휘말렸다" has nothing
    # actionable to do with loc_검은_염소_주점 itself, which isn't wrong or
    # stale in any way. The thing that's actually now inconsistent with the
    # edited fact, and the only thing there's anything to actually do about
    # (edit or delete it), is event_데이비드_2080 itself — the record describing
    # exactly what's being removed). The relevance judgment step already
    # filters out noise (an unrelated event stays unrelated); the bug was
    # that events were never even candidates to begin with, so the
    # obviously-relevant one could never be picked regardless.
    # A list, not a set — get_events_for_entity is already year-sorted, and
    # throwing that away for no reason was one half of why the final
    # displayed order ended up arbitrary (see _judge_relevance for the
    # other half).
    own_event_ids = [record["id"] for record in storage.get_events_for_entity(entity_id)]

    if field_name == "notes":
        diff = _notes_diff(previous_value, new_value)
        focus = f'"{diff}"라는 서술과 관련되거나 모순될 수 있는 내용'
        candidates = [(eid, _candidate_text(eid, include_fields=True)) for eid in own_event_ids]
    elif _is_age_field(category, field_name):
        focus = _AGE_FOCUS_DESCRIPTION
        candidates = [(eid, _candidate_text(eid, include_fields=False)) for eid in own_event_ids]
    else:
        focus = f'"{previous_value}" 또는 "{new_value}"와(과) 관련되거나 전제로 하는 규칙/서술'
        candidates = [(eid, _candidate_text(eid, include_fields=False)) for eid in own_event_ids]

    relevant_ids = _judge_relevance(candidates, focus)
    # relevant_ids is already in the model's own most-relevant-first order
    # (see _judge_relevance) — iterate it directly rather than filtering
    # `candidates` in its own (unrelated) order, which is what silently
    # discarded that ranking before.
    text_by_id = dict(candidates)
    return [
        RelevantMatch(entity_id=eid, reason=text_by_id[eid])
        for eid in relevant_ids
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
