"""RAG cross-checks (Step 4) — Phase 3.

Three probabilistic checks (rule violation, notes conflict, status
consistency), all reasoned about by the LLM since — unlike Phase 1's hard
checks — there's no deterministic ground truth here. Every Judgment carries
a human-readable `reason` so Phase 5's confirmation popup can show it as-is.
Uses the reasoning-tier model (config.get_model("reasoning")).
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import config, schema, storage


@dataclass
class Judgment:
    type: str  # "rule_violation" | "notes_conflict" | "conflict" | "clears_status"
    reason: str
    confidence: float | None = None
    entity_id: str | None = None
    status_effect_id: str | None = None


def _get_llm(tier: str = "reasoning"):
    return config.get_chat_model(tier, temperature=0)


def _invoke_llm(prompt: str, tier: str = "reasoning") -> str:
    response = _get_llm(tier).invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾을 수 없습니다: {raw!r}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# 2-1. RAG retrieval
# ---------------------------------------------------------------------------

def retrieve_context(entities: list, raw_text: str, top_k: int = 3) -> list:
    entity_texts = []
    for entity_id in entities:
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        record = storage.get_entity(category, entity_id)
        if record and record.get("notes"):
            entity_texts.append(str(record["notes"]))

    query_text = " ".join(entity_texts + [raw_text]) if entity_texts else raw_text
    results = storage.query_chroma(query_text, top_k=top_k)
    documents = results.get("documents") or [[]]
    return documents[0] if documents else []


def _get_hard_rule_texts() -> list:
    """All hard_rule=true system entries, fetched directly rather than via
    embedding similarity — the world-rule corpus is small and canonical, so
    rule-violation checks shouldn't depend on retrieval quality."""
    conn = storage.get_connection()
    storage.init_db(conn)
    rows = conn.execute('SELECT * FROM "system" WHERE hard_rule = 1').fetchall()
    conn.close()
    return [str(row["notes"]) for row in rows if row["notes"]]


# ---------------------------------------------------------------------------
# 2-2/2-3 shared: an involved entity's own stored context
# ---------------------------------------------------------------------------

_FIELD_SUMMARY_EXCLUDED = {"id", "name", "notes", "event_ids", "lifespan_check_ack"}


def entity_field_summary(record: dict) -> str | None:
    """A flat "field=value" summary of an entity's own stored fields (Phase
    10 patch 5) — gender, race, etc. `notes` and internal bookkeeping fields
    (event_ids, lifespan_check_ack) are excluded; notes gets its own line
    below, and the rest aren't domain content an LLM should reason about."""
    parts = [
        f"{name}={value}"
        for name, value in record.items()
        if name not in _FIELD_SUMMARY_EXCLUDED and value not in (None, "", [])
    ]
    return ", ".join(parts) if parts else None


def _duration_range_note(related_event: dict) -> str | None:
    """A short '(실제 기간: 2085년~2090년, 이미 종료됨)' fact for a duration-type
    related event, computed directly from its stored start_year/end_year —
    shown regardless of what the record's own `notes` text says.

    `notes` is free prose written once (by a person or the LLM) and never
    re-synced when a field like end_year is edited afterward independently
    (e.g. through the GUI field editor) — a record seeded as "2085년부터
    수감된 상태다" stays exactly that text forever even after its end_year
    is set to 2090 later. A context builder that only ever surfaces notes
    text has no way to know the record was actually closed at all, and an
    LLM asked to draft what happens next has nothing to contradict, so it
    invents its own answer instead of reading a real one that was already
    on file (caught in practice: Creator drafting a brand new "released in
    2086" event for a status whose real, already-set end_year was 2090).
    Appending the actual field values as plain text closes that gap
    without depending on notes ever being kept in sync."""
    if not related_event.get("entity"):
        return None
    start = related_event.get("start_year")
    end = related_event.get("end_year")
    if start is None and end is None:
        return None
    start_label = start if start is not None else "?"
    if end is not None:
        return f"(실제 기간: {start_label}년~{end}년, 이미 종료됨)"
    return f"(실제 기간: {start_label}년~, 현재까지 진행 중)"


def _status_effect_notes_map() -> dict:
    """{predicate_id: notes} for every status_effects.yaml entry that has
    notes set — not cached beyond schema.load_status_effects' own
    lru_cache, so an edit via the GUI editor is picked up immediately."""
    return {s["id"]: s["notes"] for s in schema.load_status_effects() if s.get("notes")}


def _duration_activity_annotation(related_event: dict, event_year: int) -> str | None:
    """[활성] tag for a duration-type related event (has a start_year) that
    is actually in effect at event_year, computed the same way
    storage.get_current_state does — Phase 10 patch 15, A: a duration
    record's notes ("2010년에 추방당했다") read as an unconditional current
    fact to an LLM with no year sense of its own, so a 2009 event (before
    the exile even started) was judged against a status that, at that point
    in the story, hadn't happened yet.

    Phase 10 patch 21: the inverse [비활성] tag (plus a prompt instruction
    telling the LLM not to treat a [비활성]-tagged record as contradiction
    grounds) was removed — it correctly suppressed false positives against
    a status/relationship that hadn't started yet, but the same suppression
    also silently defeated a genuinely load-bearing case: "미라와 쟝은
    2079년에 처음 만났다" (start_year=2079) is a boundary fact that rules
    out anything involving both of them before 2079, and got waved off as
    "그 사건 시점에는 적용되지 않는다" for a 2050 input that directly
    violated it. Not-yet-started/already-ended duration records go back to
    being plain, untagged context now — full weight, same as any other
    fact — and only a genuinely active status gets the (harmless,
    confirmatory) [활성] tag.

    Returns None for a point event (no start_year — always relevant,
    nothing to gate), when there's no event_year to compare against, or
    when the record isn't currently active."""
    start = related_event.get("start_year")
    if start is None or event_year is None:
        return None
    end = related_event.get("end_year")
    is_active = start <= event_year and (end is None or event_year <= end)
    if not is_active:
        return None
    span = f"{start}~{end if end is not None else '현재'}"
    return f"[활성, 기간 {span}, 이 사건({event_year}년) 기준]"


def _entity_context_lines(
    entities: list,
    event_year: int | None = None,
    extra_context: list | None = None,
    closed_predicates: set | None = None,
) -> list:
    """Each involved entity's own stored fields + notes (including any
    self-declared exception) + its race's notes + its related events' notes
    — the exact context `check_notes_conflict` has built since Phase 10
    patch 5. Shared with `check_rule_violation` (Phase 10 patch 10, B):
    world-rule judgment needs the same "does this entity already satisfy
    the condition, or carry an explicit exception" context, or an
    already-saved qualifying value (e.g. a mana circle count) and a
    self-declared exception (e.g. "체질상 마나 서클 없이도 마법 가능") both go
    invisible the moment the current sentence doesn't restate them.

    `event_year`, when given, gets each duration-type related event that's
    actually in effect at that year tagged [활성] (Phase 10 patch 15, A,
    narrowed by patch 21) — confirmatory context, not a filter; a record
    with no tag (not currently active, or a point event) still carries its
    full weight in the judgment.

    Phase 10 patch 18 briefly dropped point events dated *after* event_year
    from context entirely, on the theory that a point event which "hasn't
    happened yet" (relative to what's being checked) can't be the source of
    a contradiction. Rolled back (patch 21) — that's only true when the
    later point event is causally independent of the earlier one. It's
    false for a boundary fact: "쟝과 미라가 2079년에 처음 만났다" (a point
    event) *constrains* what could have happened earlier, and inserting
    "2050년에 쟝과 미라가 함께 놀았다" went through unchallenged because the
    2079 meeting was excluded from context for being "in the future" of the
    2050 event being checked — exactly backwards. Every point event's notes
    stay in context regardless of year again; only duration records get the
    patch 15 annotate-don't-drop treatment.

    `extra_context` (Phase 10 patch 22) lets a caller append facts that
    aren't in storage yet — specifically Creator, which validates a
    multi-event draft sequentially before anything is saved, and needs
    event 3's check to see events 1-2 as real context even though they're
    still only in memory. Every existing caller passes nothing, so this is
    additive-only for the normal chat pipeline.

    `closed_predicates` (Phase 10 patch 22 follow-up): a set of
    (entity_id, predicate) pairs that a caller knows have already been
    closed by an earlier-in-this-same-batch event, even though storage
    itself still shows the record open (nothing's saved yet). Without this,
    a still-open DB record kept getting its confirmatory [활성] tag even
    after Creator had already drafted the `clear` event ending it earlier in
    the same draft — the stale tag directly told the LLM the status was
    still active, outweighing the plain-text mention of the release sitting
    in extra_context. Matching records are annotated as if not active
    (patch 21's plain, untagged treatment), not dropped — the fact itself
    still carries full weight, just without the misleading confirmatory tag.

    Deduped by event id (token-diet pass): two involved entities that share
    most of their history used to each print every shared event in full
    under their own section, doubling that portion of the context for zero
    added information — see creator._entity_context_block, which had the
    exact same duplication and got the same fix. Fixed a latent
    correctness gap while restructuring this: `already_closed` used to be
    checked as (currently-iterated entity_id, predicate), which only
    actually matches what closed_predicates stores (always the record's
    "entity"/closer side, from duration_effect.get("entity")) when that
    tagged entity happens to be iterated from the entity side — from the
    target side it silently never matched, so a record already closed
    earlier in the same draft could still show its stale [활성] tag
    whenever it was reached via the target's own event list instead of the
    closer's. Keying off the record's own `entity` field directly fixes
    that regardless of which tagged entity is being walked."""
    lines = []
    seen_events: dict = {}
    event_order = []
    for entity_id in entities:
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        record = storage.get_entity(category, entity_id)
        if not record:
            continue
        summary = entity_field_summary(record)
        if summary:
            lines.append(f"{entity_id}의 저장된 정보: {summary}")
        if record.get("notes"):
            lines.append(f"{entity_id}: {record['notes']}")
        # A character's race carries its own notes (e.g. dietary restrictions)
        # that the character itself doesn't repeat.
        if category == "character" and record.get("race"):
            race_record = storage.get_entity("race", record["race"])
            if race_record and race_record.get("notes"):
                lines.append(f"{record['race']}: {race_record['notes']}")
        # Phase 10 patch 2 (B): a constraint often lives only in an event's
        # notes (e.g. a faction's founding record saying "여성만 가입
        # 가능"), never on the entity's own `notes` field — since
        # `relationship` was retired, event_ids/get_events_for_entity is the
        # only remaining path to that text, and this check never read it.
        for related_event in storage.get_events_for_entity(entity_id):
            range_note = _duration_range_note(related_event)
            if not related_event.get("notes") and not range_note:
                continue
            eid = related_event["id"]
            if eid not in seen_events:
                seen_events[eid] = {"entities": [], "event": related_event, "range_note": range_note}
                event_order.append(eid)
            if entity_id not in seen_events[eid]["entities"]:
                seen_events[eid]["entities"].append(entity_id)

    for eid in event_order:
        info = seen_events[eid]
        related_event = info["event"]
        record_entity = related_event.get("entity")
        already_closed = bool(
            closed_predicates and record_entity
            and (record_entity, related_event.get("predicate")) in closed_predicates
        )
        annotation = None if already_closed else _duration_activity_annotation(related_event, event_year)
        prefix = f"{annotation} " if annotation else ""
        entity_label = ", ".join(info["entities"])
        line = f"{entity_label}의 관련 기록({eid}): {prefix}{related_event.get('notes') or ''}"
        if info["range_note"]:
            line += f" {info['range_note']}"
        # Phase 10 patch 22 follow-up 3: the predicate's own registered
        # meaning (status_effects.yaml's notes field), when set, is
        # appended here regardless of active/inactive — this is what
        # actually told the LLM "using an item requires currently
        # owning it" or "imprisoned means physically unable to leave",
        # instead of the LLM inferring real-world implications from a
        # bare predicate name alone.
        predicate_notes = _status_effect_notes_map().get(related_event.get("predicate"))
        if predicate_notes:
            line += f" (predicate '{related_event.get('predicate')}'의 의미: {predicate_notes})"
        lines.append(line)
    if extra_context:
        lines.extend(extra_context)
    return lines


# ---------------------------------------------------------------------------
# 2-2. World-rule violation
# ---------------------------------------------------------------------------

def check_rule_violation(
    entities: list,
    raw_text: str,
    context_docs: list,
    event_year: int | None = None,
    extra_context: list | None = None,
    closed_predicates: set | None = None,
) -> Judgment | None:
    docs = "\n".join(f"- {d}" for d in context_docs) if context_docs else "(관련 규칙 없음)"
    context_lines = _entity_context_lines(entities, event_year, extra_context, closed_predicates)
    entity_context = (
        "\n".join(f"- {line}" for line in context_lines)
        if context_lines else "(참고할 엔티티 정보 없음)"
    )
    print(f"[rag_check] check_rule_violation 엔티티 컨텍스트:\n{entity_context}")
    prompt = (
        "너는 판타지 세계관의 규칙 감사관이다. 아래는 이 세계관에서 확정된 규칙 및 관련 문서다. "
        "새로 입력된 사건 문장이 이 규칙을 위반하는지 판단하라.\n\n"
        f"세계관 규칙/문서:\n{docs}\n\n"
        f"관여 엔티티의 기존 저장 정보(자기 예외 조항 포함):\n{entity_context}\n\n"
        f"사건 문장: {raw_text}\n\n"
        "관련 기록 중 [활성] 표시가 붙은 것은 이 사건의 연도를 기준으로 그 상태/관계가 실제로 "
        "유효함을 뜻하는 확인 정보다 — 표시가 없다고 해서 그 기록을 무시해도 된다는 뜻은 "
        "아니며, 다른 모든 기록과 동일하게 그 내용 그대로 판단에 반영하라.\n\n"
        "규칙이 특정 전제조건(도구, 자격, 재료 등)을 요구하는데, 사건 문장에 그 전제조건이 "
        "충족되었다는 언급이 전혀 없다면 — 굳이 결여를 명시하지 않았더라도 — 위반 가능성이 "
        "있는 것으로 판단하라. 단, 위 '관여 엔티티의 기존 저장 정보'에 그 전제조건이 이미 "
        "충족되어 있음을 보여주는 값이 있거나(예: 필요한 자원을 이미 보유한 것으로 저장됨), "
        "해당 엔티티에게 적용되는 명시적 예외 조항이 있다면 — 사건 문장에 재언급이 없어도 — "
        "그 정보를 근거로 위반이 아니라고 판단하라. 규칙이 금지하는 행위 자체가 문장에 "
        "등장하고, 기존 저장 정보에도 전제조건 충족이나 예외를 뒷받침할 근거가 전혀 없다면 "
        "위반 쪽으로 판단하는 것이 기본값이다.\n\n"
        "규칙에 상관관계(정도-비례) 조항이 있는 경우 — 예: \"많을수록 강하다\" 같이 수치와 "
        "어떤 속성이 비례하는 규칙 — 판단은 \"최소치를 충족했는가\"가 아니라 전체 범위에서 "
        "상대적으로 어느 위치인가로 하라. 예를 들어 범위가 1~10일 때 2나 3은 최소치(1)는 "
        "아니지만 여전히 하위권이라, \"매우 강력하다\"/\"엄청나게 강력하다\" 같은 극단적으로 "
        "높은 평가와는 여전히 안 어울린다. 사건 문장이나 관여 엔티티의 저장 정보에 스스로 "
        "주장하는 평판·능력 서술이 있다면, 그 서술이 자신이 가진 수치와 상관관계상 앞뒤가 "
        "맞는지 반드시 확인하라. 단, 규칙 자체에 \"보통 도달하기 어렵다\", \"대부분 OO 수준에 "
        "머문다\" 같은 난이도·분포에 대한 예외·완화 조항이 있다면, 단순 절대 수치가 아니라 "
        "그 조항이 암시하는 실제 난이도를 기준으로 판단하라 — 그런 예외 조항이 없을 때만 "
        "범위 내 위치로 대략 판단하라.\n\n"
        "규칙을 위반할 가능성이 있으면:\n"
        '{"violation": true, "reason": "위반이라고 판단한 근거", "confidence": 0.0에서 1.0 사이 숫자}\n'
        "위반 가능성이 없으면:\n"
        '{"violation": false}\n'
        "위 JSON 형식으로만 답하고 다른 텍스트는 출력하지 마라."
    )

    raw = _invoke_llm(prompt)
    data = _extract_json(raw)
    if not data.get("violation"):
        print("[rag_check] check_rule_violation: 위반 없음")
        return None

    print(f"[rag_check] check_rule_violation: 위반 감지 — {data.get('reason', '')}")
    return Judgment(
        type="rule_violation",
        reason=data.get("reason", ""),
        confidence=data.get("confidence"),
    )


# ---------------------------------------------------------------------------
# 2-3. Notes-based qualitative conflict
# ---------------------------------------------------------------------------

def check_notes_conflict(
    entities: list,
    raw_text: str,
    event_year: int | None = None,
    extra_context: list | None = None,
    closed_predicates: set | None = None,
) -> Judgment | None:
    notes_lines = _entity_context_lines(entities, event_year, extra_context, closed_predicates)

    if not notes_lines:
        print("[rag_check] check_notes_conflict: 참고할 notes가 없어 LLM 호출 생략")
        return None

    notes_block = "\n".join(f"- {line}" for line in notes_lines)
    print(f"[rag_check] check_notes_conflict 컨텍스트:\n{notes_block}")
    prompt = (
        "너는 판타지 세계관의 설정 감사관이다. 아래는 관련 엔티티들의 기존 설정(notes)이다. "
        "새로 입력된 사건 문장이 이 설정과 모순되는지 판단하라.\n\n"
        "관련 기록 중 [활성] 표시가 붙은 것은 이 사건의 연도를 기준으로 그 상태/관계가 실제로 "
        "유효함을 뜻하는 확인 정보다 — 표시가 없다고 해서 그 기록을 무시해도 된다는 뜻은 "
        "아니며, 다른 모든 기록과 동일하게 그 내용 그대로 판단에 반영하라.\n\n"
        "명시적 규칙 위반뿐 아니라, 서술된 성격·위험도·상관관계와 행동/속성 사이의 모순도 "
        "확인하라:\n"
        "1. 관련 엔티티(장소, 사물, 시스템 등)의 notes/규칙에 성격·용도·제약·상관관계를 "
        "규정하는 서술이 있는지 확인하라. 규정하는 서술의 예:\n"
        "   - 위험도/성격 (\"목숨이 위험하다\", \"결투를 하는 곳이다\")\n"
        "   - 접근 제약 (\"출입 금지\", \"선택받은 자만\")\n"
        "   - 효과 강도 (\"일상생활이 불가능하다\")\n"
        "   - 상관관계 (\"많을수록 강하다\", \"적을수록 약하다\" 같은 정도-비례 규칙)\n"
        "2. 이런 서술이 있으면, 새로 입력된 행동이나 다른 엔티티의 자기 서술이 그 규정의 "
        "통상적 함의와 정면으로 반대되는지 판단하라.\n"
        "   - 위험/제약이 명시된 대상에서 안전하고 여유로운 행동(피크닉, 산책, 낮잠 등)을 "
        "하면 -> 모순 후보\n"
        "   - 반대로 안전하다고 명시된 대상에서 위험하거나 폭력적인 사건이 발생하면 -> 모순 "
        "후보\n"
        "   - 상관관계 규칙이 있는 경우, 한 엔티티가 스스로 주장하는 속성(평판, 능력 등)이 그 "
        "상관관계상 자신이 가진 다른 값과 앞뒤가 맞는지도 확인하라. 판단은 \"최소치인가 "
        "아닌가\"가 아니라 전체 범위/맥락에서 상대적으로 어느 위치인가로 하라 — 예를 들어 "
        "범위가 1~10일 때 2는 최소치(1)는 아니지만 여전히 하위권이라, \"매우 강력하다\" 같은 "
        "극단적으로 높은 평가와는 여전히 안 어울린다. 단, 범위를 기계적으로 균등(선형)하다고 "
        "가정하지 마라 — 규칙 서술에 구간별 추가 단서(예: \"5개부터 강하다\", \"10개는 "
        "지금까지 나타난 적이 없다\")가 있으면 그 단서를 우선 근거로 상대적 위치를 판단하고, "
        "그런 단서가 전혀 없을 때만 범위 내 위치로 대략 판단하라\n"
        "3. 사건 문장이 특정 인물이 특정 사물(무기, 도구, 유물 등)을 사용·착용·소지·휘두르는 "
        "등 실제로 다루는 행동을 서술한다면, 이는 그 시점에 그 사물을 소유·보유하고 있다는 "
        "주장으로 간주하라. 관련 기록에 그 사물의 소유 관계 기록(예: '...년부터 ...년까지 "
        "소유했으나 ...년에 잃어버렸다/도난당했다/파괴되었다')이 있는데 사건 문장의 시점이 그 "
        "소유가 유효한 기간 밖이라면, 이는 명시적 소속·자격 주장(예: '탈퇴한 단체 소속으로서 "
        "행동')과 동일한 무게의 모순 후보로 판단하라 — 소유 사실이 행동 동사에만 암시되어 "
        "있다는 이유로 가볍게 넘기지 마라.\n"
        "4. 단, 다음의 경우 모순으로 보지 않는다:\n"
        "   - 행동이 발생한 위치나 맥락이 그 규정이 적용되는 범위 밖임이 문장에 명시된 경우 "
        "(예: \"결투를 하는 투기장\"이어도, 행동이 \"관중석\"처럼 위험 구역과 구분되는 별도 "
        "장소에서 일어났다면 자연스러운 상황일 수 있다)\n"
        "   - 행동 자체가 이미 그 위험/제약을 인지하고 대응하는 것으로 보이는 경우(전투, "
        "경계, 도주 등)\n"
        "5. 판단이 애매하면 확신 없이도 모순 가능성이 있다고 보고하라 — 최종 판단은 사람이 "
        "확인 후 결정하므로, 놓치는 것보다 애매하게라도 짚어주는 쪽이 낫다. 확신이 없다는 걸 "
        "reason에 명시해도 된다.\n\n"
        f"기존 설정:\n{notes_block}\n\n"
        f"사건 문장: {raw_text}\n\n"
        "모순 가능성이 있으면:\n"
        '{"conflict": true, "reason": "모순이라고 판단한 근거"}\n'
        "없으면:\n"
        '{"conflict": false}\n'
        "위 JSON 형식으로만 답하고 다른 텍스트는 출력하지 마라."
    )

    raw = _invoke_llm(prompt)
    data = _extract_json(raw)
    if not data.get("conflict"):
        print("[rag_check] check_notes_conflict: 모순 없음")
        return None

    print(f"[rag_check] check_notes_conflict: 모순 감지 — {data.get('reason', '')}")
    return Judgment(type="notes_conflict", reason=data.get("reason", ""))


# ---------------------------------------------------------------------------
# 2-3b. Combined rule-violation + notes-conflict (Phase 10 patch 18)
# ---------------------------------------------------------------------------

def check_rule_and_notes(
    entities: list,
    raw_text: str,
    context_docs: list,
    event_year: int | None = None,
    extra_context: list | None = None,
    closed_predicates: set | None = None,
) -> list:
    """check_rule_violation and check_notes_conflict, combined into one LLM
    call instead of two. Both checks were independently pulling the exact
    same _entity_context_lines() context and paying for it in separate
    reasoning-tier requests — same reasoning, same data, twice the token
    cost, every single input. This builds that context once, asks for both
    judgments in one JSON response, and returns whichever of the two fired
    (0, 1, or 2 Judgments) — the per-check reasoning criteria are kept
    completely separate within the prompt (not blended), so this changes
    *cost*, not *what* gets judged or how. check_rule_violation/
    check_notes_conflict themselves are untouched and still usable standalone
    (tests call them directly) — only the integration points below switch to
    this."""
    docs = "\n".join(f"- {d}" for d in context_docs) if context_docs else "(관련 규칙 없음)"
    context_lines = _entity_context_lines(entities, event_year, extra_context, closed_predicates)
    entity_context = (
        "\n".join(f"- {line}" for line in context_lines)
        if context_lines else "(참고할 엔티티 정보 없음)"
    )
    print(f"[rag_check] check_rule_and_notes 컨텍스트:\n{entity_context}")

    prompt = (
        "너는 판타지 세계관의 규칙·설정 감사관이다. 새로 입력된 사건 문장에 대해 아래 두 가지를 "
        "각각 독립적으로 판단하라 — 하나가 위반/모순이 아니라고 다른 하나까지 그런 것은 아니다.\n\n"
        f"세계관 규칙/문서:\n{docs}\n\n"
        f"관여 엔티티의 기존 저장 정보 및 관련 기록(자기 예외 조항 포함):\n{entity_context}\n\n"
        f"사건 문장: {raw_text}\n\n"
        "관련 기록 중 [활성] 표시가 붙은 것은 이 사건의 연도를 기준으로 그 상태/관계가 실제로 "
        "유효함을 뜻하는 확인 정보다 — 표시가 없다고 해서 그 기록을 무시해도 된다는 뜻은 "
        "아니며, 다른 모든 기록과 동일하게 그 내용 그대로 판단에 반영하라. 단, 사건 문장 "
        "자체가 바로 그 [활성] 상태/관계를 끝내는 행동(해제, 석방, 사면, 탈출, 해체, 종료 등)을"
        "서술한다면, 그 활성 상태와 공존한다고 해서 모순/위반으로 보지 마라 — 오히려 그 상태가 "
        "그 시점까지 유효했기 때문에 끝낼 수 있는 것이며, 이런 종료 서술은 그 활성 기록과 "
        "정면으로 맞아떨어지는 정상적인 사건이다. (그 상태를 끝내는 것과 무관한 다른 이유로 "
        "위반/모순이 있다면 그건 별개로 정상 판단하라.)\n\n"
        "=== 판단 1: 세계관 규칙 위반 여부 ===\n\n"
        "규칙이 특정 전제조건(도구, 자격, 재료 등)을 요구하는데, 사건 문장에 그 전제조건이 "
        "충족되었다는 언급이 전혀 없다면 — 굳이 결여를 명시하지 않았더라도 — 위반 가능성이 "
        "있는 것으로 판단하라. 단, 위 '관여 엔티티의 기존 저장 정보'에 그 전제조건이 이미 "
        "충족되어 있음을 보여주는 값이 있거나(예: 필요한 자원을 이미 보유한 것으로 저장됨), "
        "해당 엔티티에게 적용되는 명시적 예외 조항이 있다면 — 사건 문장에 재언급이 없어도 — "
        "그 정보를 근거로 위반이 아니라고 판단하라. 규칙이 금지하는 행위 자체가 문장에 "
        "등장하고, 기존 저장 정보에도 전제조건 충족이나 예외를 뒷받침할 근거가 전혀 없다면 "
        "위반 쪽으로 판단하는 것이 기본값이다.\n\n"
        "규칙에 상관관계(정도-비례) 조항이 있는 경우 — 예: \"많을수록 강하다\" 같이 수치와 "
        "어떤 속성이 비례하는 규칙 — 판단은 \"최소치를 충족했는가\"가 아니라 전체 범위에서 "
        "상대적으로 어느 위치인가로 하라. 예를 들어 범위가 1~10일 때 2나 3은 최소치(1)는 "
        "아니지만 여전히 하위권이라, \"매우 강력하다\"/\"엄청나게 강력하다\" 같은 극단적으로 "
        "높은 평가와는 여전히 안 어울린다. 사건 문장이나 관여 엔티티의 저장 정보에 스스로 "
        "주장하는 평판·능력 서술이 있다면, 그 서술이 자신이 가진 수치와 상관관계상 앞뒤가 "
        "맞는지 반드시 확인하라. 단, 규칙 자체에 \"보통 도달하기 어렵다\", \"대부분 OO 수준에 "
        "머문다\" 같은 난이도·분포에 대한 예외·완화 조항이 있다면, 단순 절대 수치가 아니라 "
        "그 조항이 암시하는 실제 난이도를 기준으로 판단하라 — 그런 예외 조항이 없을 때만 "
        "범위 내 위치로 대략 판단하라.\n\n"
        "=== 판단 2: 설정(notes) 모순 여부 ===\n\n"
        "명시적 규칙 위반뿐 아니라, 서술된 성격·위험도·상관관계와 행동/속성 사이의 모순도 "
        "확인하라:\n"
        "1. 관련 엔티티(장소, 사물, 시스템 등)의 notes/규칙에 성격·용도·제약·상관관계를 "
        "규정하는 서술이 있는지 확인하라. 규정하는 서술의 예:\n"
        "   - 위험도/성격 (\"목숨이 위험하다\", \"결투를 하는 곳이다\")\n"
        "   - 접근 제약 (\"출입 금지\", \"선택받은 자만\")\n"
        "   - 효과 강도 (\"일상생활이 불가능하다\")\n"
        "   - 상관관계 (\"많을수록 강하다\", \"적을수록 약하다\" 같은 정도-비례 규칙)\n"
        "2. 이런 서술이 있으면, 새로 입력된 행동이나 다른 엔티티의 자기 서술이 그 규정의 "
        "통상적 함의와 정면으로 반대되는지 판단하라.\n"
        "   - 위험/제약이 명시된 대상에서 안전하고 여유로운 행동(피크닉, 산책, 낮잠 등)을 "
        "하면 -> 모순 후보\n"
        "   - 반대로 안전하다고 명시된 대상에서 위험하거나 폭력적인 사건이 발생하면 -> 모순 "
        "후보\n"
        "   - 상관관계 규칙이 있는 경우, 한 엔티티가 스스로 주장하는 속성(평판, 능력 등)이 그 "
        "상관관계상 자신이 가진 다른 값과 앞뒤가 맞는지도 확인하라(위 판단 1과 동일한 상대적 "
        "위치 기준을 적용하라).\n"
        "3. 사건 문장이 특정 인물이 특정 사물(무기, 도구, 유물 등)을 사용·착용·소지·휘두르는 "
        "등 실제로 다루는 행동을 서술한다면, 이는 그 시점에 그 사물을 소유·보유하고 있다는 "
        "주장으로 간주하라. 관련 기록에 그 사물의 소유 관계 기록(예: '...년부터 ...년까지 "
        "소유했으나 ...년에 잃어버렸다/도난당했다/파괴되었다')이 있는데 사건 문장의 시점이 그 "
        "소유가 유효한 기간 밖이라면, 이는 명시적 소속·자격 주장(예: '탈퇴한 단체 소속으로서 "
        "행동')과 동일한 무게의 모순 후보로 판단하라 — 소유 사실이 행동 동사에만 암시되어 "
        "있다는 이유로 가볍게 넘기지 마라.\n"
        "4. 단, 다음의 경우 모순으로 보지 않는다:\n"
        "   - 행동이 발생한 위치나 맥락이 그 규정이 적용되는 범위 밖임이 문장에 명시된 경우 "
        "(예: \"결투를 하는 투기장\"이어도, 행동이 \"관중석\"처럼 위험 구역과 구분되는 별도 "
        "장소에서 일어났다면 자연스러운 상황일 수 있다)\n"
        "   - 행동 자체가 이미 그 위험/제약을 인지하고 대응하는 것으로 보이는 경우(전투, "
        "경계, 도주 등)\n"
        "5. 판단이 애매하면 확신 없이도 모순 가능성이 있다고 보고하라 — 최종 판단은 사람이 "
        "확인 후 결정하므로, 놓치는 것보다 애매하게라도 짚어주는 쪽이 낫다. 확신이 없다는 걸 "
        "reason에 명시해도 된다.\n\n"
        "아래 JSON 형식으로만 답하라 (다른 텍스트 금지):\n"
        "{\n"
        '  "rule_violation": {"violation": true 또는 false, "reason": "위반 시에만", '
        '"confidence": "위반 시에만, 0.0~1.0"},\n'
        '  "notes_conflict": {"conflict": true 또는 false, "reason": "모순 시에만"}\n'
        "}\n"
    )

    raw = _invoke_llm(prompt)
    data = _extract_json(raw)
    judgments = []

    rule_data = data.get("rule_violation") or {}
    if rule_data.get("violation"):
        print(f"[rag_check] check_rule_and_notes: 규칙 위반 감지 — {rule_data.get('reason', '')}")
        judgments.append(
            Judgment(type="rule_violation", reason=rule_data.get("reason", ""), confidence=rule_data.get("confidence"))
        )
    else:
        print("[rag_check] check_rule_and_notes: 규칙 위반 없음")

    notes_data = data.get("notes_conflict") or {}
    if notes_data.get("conflict"):
        print(f"[rag_check] check_rule_and_notes: 설정 모순 감지 — {notes_data.get('reason', '')}")
        judgments.append(Judgment(type="notes_conflict", reason=notes_data.get("reason", "")))
    else:
        print("[rag_check] check_rule_and_notes: 설정 모순 없음")

    return judgments


# ---------------------------------------------------------------------------
# 2-4. Reversible status consistency
# ---------------------------------------------------------------------------

def check_status_consistency(entities: list, raw_text: str, event_year: int) -> list[Judgment]:
    """Batched across every entity involved in the same event (Phase 10
    token-diet pass) — this used to run once per entity_id, each its own
    separate LLM call, so a 2-character event already cost 2 of these on
    top of check_rule_and_notes' own call. Same judgment logic and prompt
    style, just asked once for every entity that actually has an active
    individual status, instead of once per entity regardless.

    Gated per-entity by event_year: if none of an entity's personal-status
    duration events (Phase 10 — a timeline record with predicate=a
    status_effects.yaml id, no target) actually cover event_year, there's
    nothing for the event to be consistent or inconsistent *with* at that
    point in time — an entity with no active status is simply left out of
    the prompt entirely, and if nobody involved has one, the LLM call is
    skipped altogether rather than asking it to judge against a status
    that (from the timeline's perspective) hadn't started yet, or had
    already ended, when this event happened."""
    all_effects = schema.load_status_effects()
    # Phase 10 patch 16: status_effects.yaml now also holds target-bearing
    # relational predicates (exiled, ...) alongside personal statuses — this
    # check is specifically about the latter (its prompt has no notion of a
    # target to reason about), so relational entries are excluded here, not
    # just historically absent.
    status_ids = [s["id"] for s in all_effects if s.get("type", "individual") == "individual"]
    label_map = {s["id"]: s["label"] for s in all_effects}
    notes_map = {s["id"]: s.get("notes") for s in all_effects}

    per_entity_active = {}
    for entity_id in entities:
        active = [sid for sid in status_ids if storage.get_current_state(entity_id, sid, event_year)]
        if active:
            per_entity_active[entity_id] = active
    if not per_entity_active:
        return []

    def _effect_lines(active_effects: list) -> str:
        return "\n".join(
            f"  - {eid} ({label_map.get(eid, eid)})" + (f": {notes_map[eid]}" if notes_map.get(eid) else "")
            for eid in active_effects
        )

    entities_block = "\n".join(
        f"- {entity_id}:\n{_effect_lines(active)}" for entity_id, active in per_entity_active.items()
    )

    prompt = (
        "너는 판타지 세계관의 상태 정합성 감사관이다. 아래 엔티티들에게 각각 현재 해제되지 "
        "않은 상태(reversible status)가 걸려 있다.\n\n"
        f"엔티티별 현재 상태:\n{entities_block}\n\n"
        "각 상태 옆에 붙은 설명(있는 경우)은 그 상태가 실제로 무엇을 허용하고 무엇을 "
        "금지하는지 정의한 것이다 — 이름만으로 짐작하지 말고, 설명이 있다면 그 내용을 "
        "판단의 직접적인 근거로 삼아라. 설명이 없는 상태는 이름과 상식적인 함의로 판단하라. "
        "엔티티별로 서로 독립적으로 판단하라 — 한 엔티티의 상태 판단이 다른 엔티티의 판단에 "
        "영향을 주지 않는다.\n\n"
        f"새로 입력된 사건 문장: {raw_text}\n\n"
        "이 문장이 위 엔티티 각각의 상태와 어떤 관계인지 판단하라. 자연스럽게 양립하는 "
        "엔티티는 results 배열에서 아예 제외하고, 그 외의 경우만 아래 형식으로 답하라:\n"
        '{"results": [{"entity_id": "해당 엔티티", "result": "clears" 또는 "conflict", '
        '"status_effect_id": "해당 상태 id", "reason": "판단 근거"}, ...]}\n'
        '모든 엔티티가 자연스럽게 양립하면 {"results": []}을 반환하라. JSON 이외의 텍스트는 '
        "출력하지 마라."
    )

    # "simple" tier (not "reasoning", unlike every other check in this
    # file) — this judgment is narrower and more mechanical than a rule/
    # notes-conflict call: compare a sentence against an explicitly
    # described status and decide ok/clears/conflict, no open-ended world-
    # rule reasoning involved. Worth testing against real status-heavy
    # scenes (imprisoned, incapacitated, ...) before trusting it broadly —
    # a weaker model could plausibly miss a subtler case.
    raw = _invoke_llm(prompt, tier="simple")
    data = _extract_json(raw)
    items = data.get("results") or []

    judgments = []
    for item in items:
        result = item.get("result")
        entity_id = item.get("entity_id")
        if result not in ("clears", "conflict") or entity_id not in per_entity_active:
            continue
        judgments.append(
            Judgment(
                type="clears_status" if result == "clears" else "conflict",
                reason=item.get("reason", ""),
                entity_id=entity_id,
                status_effect_id=item.get("status_effect_id"),
            )
        )
    return judgments


# ---------------------------------------------------------------------------
# 2-5. Integration
# ---------------------------------------------------------------------------

def run_entity_creation_checks(entities: list, raw_text: str) -> list:
    """Step 4 for a brand-new entity's directly-saved fields/notes (Phase 10
    patch 9) — a bare attribute statement like "[아마조네스 용병단]은 여성만이
    가입 가능한 용병단이다" never becomes a timeline event (no year), so it
    used to skip Step 4 entirely and reach storage unchecked. Same world-rule
    + notes-conflict judgments as run_rag_checks, minus
    check_status_consistency — that check is anchored to a specific
    event_year, which a year-less attribute creation doesn't have."""
    hard_rule_docs = _get_hard_rule_texts()
    return check_rule_and_notes(entities, raw_text, hard_rule_docs)


def run_rag_checks(entities: list, raw_text: str, event_year: int) -> list:
    print(f"[rag_check] run_rag_checks 호출: entities={entities}, year={event_year}")

    # check_rule_and_notes gets the canonical hard-rule texts plus each
    # involved entity's own stored context (Phase 10 patch 10, B) — not the
    # generic similarity-search context_docs, since mixing in unrelated
    # retrieved documents was observed to dilute the prompt enough that an
    # actual violation went undetected. retrieve_context() stays available
    # as a general-purpose utility, just not fed into this specific check.
    hard_rule_docs = _get_hard_rule_texts()

    # check_rule_and_notes and check_status_consistency are independent —
    # neither's result feeds the other — so they can run concurrently
    # instead of back to back, roughly halving wall-clock latency for
    # events where both actually fire an LLM call. Gated by
    # PARALLEL_RAG_CHECKS (default true): a cloud provider handles two
    # concurrent requests fine, but a local Ollama instance running as
    # large a model as VRAM allows may only be able to serve one
    # generation at a time, where firing two at once contends for the same
    # GPU memory instead of actually parallelizing — see
    # config.parallel_rag_checks_enabled's own docstring.
    if config.parallel_rag_checks_enabled():
        with ThreadPoolExecutor(max_workers=2) as pool:
            rule_notes_future = pool.submit(check_rule_and_notes, entities, raw_text, hard_rule_docs, event_year)
            status_future = pool.submit(check_status_consistency, entities, raw_text, event_year)
            judgments = rule_notes_future.result()
            judgments.extend(status_future.result())
    else:
        judgments = check_rule_and_notes(entities, raw_text, hard_rule_docs, event_year)
        judgments.extend(check_status_consistency(entities, raw_text, event_year))

    return judgments
