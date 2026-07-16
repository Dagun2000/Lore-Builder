"""RAG cross-checks (Step 4) — Phase 3.

Three probabilistic checks (rule violation, notes conflict, status
consistency), all reasoned about by the LLM since — unlike Phase 1's hard
checks — there's no deterministic ground truth here. Every Judgment carries
a human-readable `reason` so Phase 5's confirmation popup can show it as-is.
Uses the reasoning-tier model (config.get_model("reasoning")).
"""

import json
import re
from dataclasses import dataclass

from . import config, schema, storage


@dataclass
class Judgment:
    type: str  # "rule_violation" | "notes_conflict" | "conflict" | "clears_status"
    reason: str
    confidence: float | None = None
    entity_id: str | None = None
    status_effect_id: str | None = None


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


def _entity_context_lines(entities: list) -> list:
    """Each involved entity's own stored fields + notes (including any
    self-declared exception) + its race's notes + its related events' notes
    — the exact context `check_notes_conflict` has built since Phase 10
    patch 5. Shared with `check_rule_violation` (Phase 10 patch 10, B):
    world-rule judgment needs the same "does this entity already satisfy
    the condition, or carry an explicit exception" context, or an
    already-saved qualifying value (e.g. a mana circle count) and a
    self-declared exception (e.g. "체질상 마나 서클 없이도 마법 가능") both go
    invisible the moment the current sentence doesn't restate them."""
    lines = []
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
            if related_event.get("notes"):
                lines.append(
                    f"{entity_id}의 관련 기록({related_event['id']}): {related_event['notes']}"
                )
    return lines


# ---------------------------------------------------------------------------
# 2-2. World-rule violation
# ---------------------------------------------------------------------------

def check_rule_violation(entities: list, raw_text: str, context_docs: list) -> Judgment | None:
    docs = "\n".join(f"- {d}" for d in context_docs) if context_docs else "(관련 규칙 없음)"
    context_lines = _entity_context_lines(entities)
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
        "규칙이 특정 전제조건(도구, 자격, 재료 등)을 요구하는데, 사건 문장에 그 전제조건이 "
        "충족되었다는 언급이 전혀 없다면 — 굳이 결여를 명시하지 않았더라도 — 위반 가능성이 "
        "있는 것으로 판단하라. 단, 위 '관여 엔티티의 기존 저장 정보'에 그 전제조건이 이미 "
        "충족되어 있음을 보여주는 값이 있거나(예: 필요한 자원을 이미 보유한 것으로 저장됨), "
        "해당 엔티티에게 적용되는 명시적 예외 조항이 있다면 — 사건 문장에 재언급이 없어도 — "
        "그 정보를 근거로 위반이 아니라고 판단하라. 규칙이 금지하는 행위 자체가 문장에 "
        "등장하고, 기존 저장 정보에도 전제조건 충족이나 예외를 뒷받침할 근거가 전혀 없다면 "
        "위반 쪽으로 판단하는 것이 기본값이다.\n\n"
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

def check_notes_conflict(entities: list, raw_text: str) -> Judgment | None:
    notes_lines = _entity_context_lines(entities)

    if not notes_lines:
        print("[rag_check] check_notes_conflict: 참고할 notes가 없어 LLM 호출 생략")
        return None

    notes_block = "\n".join(f"- {line}" for line in notes_lines)
    print(f"[rag_check] check_notes_conflict 컨텍스트:\n{notes_block}")
    prompt = (
        "너는 판타지 세계관의 설정 감사관이다. 아래는 관련 엔티티들의 기존 설정(notes)이다. "
        "새로 입력된 사건 문장이 이 설정과 모순되는지 판단하라.\n\n"
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
        "상관관계상 자신이 가진 다른 값(예: 최소치의 자원량)과 앞뒤가 맞는지도 확인하라\n"
        "3. 단, 다음의 경우 모순으로 보지 않는다:\n"
        "   - 행동이 발생한 위치나 맥락이 그 규정이 적용되는 범위 밖임이 문장에 명시된 경우 "
        "(예: \"결투를 하는 투기장\"이어도, 행동이 \"관중석\"처럼 위험 구역과 구분되는 별도 "
        "장소에서 일어났다면 자연스러운 상황일 수 있다)\n"
        "   - 행동 자체가 이미 그 위험/제약을 인지하고 대응하는 것으로 보이는 경우(전투, "
        "경계, 도주 등)\n"
        "4. 판단이 애매하면 확신 없이도 모순 가능성이 있다고 보고하라 — 최종 판단은 사람이 "
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
# 2-4. Reversible status consistency
# ---------------------------------------------------------------------------

def check_status_consistency(entity_id: str, raw_text: str, event_year: int) -> Judgment | None:
    """Gated by event_year: if none of entity_id's personal-status duration
    events (Phase 10 — a timeline record with predicate=a status_effects.yaml
    id, no target) actually cover event_year, there's nothing for the event
    to be consistent or inconsistent *with* at that point in time, so skip
    the LLM call entirely rather than asking it to judge against a status
    that (from the timeline's perspective) hadn't started yet, or had
    already ended, when this event happened."""
    status_ids = [s["id"] for s in schema.load_status_effects()]
    active_effects = [
        sid for sid in status_ids if storage.get_current_state(entity_id, sid, event_year)
    ]
    if not active_effects:
        return None

    label_map = {s["id"]: s["label"] for s in schema.load_status_effects()}
    effect_lines = "\n".join(
        f"- {eid} ({label_map.get(eid, eid)})" for eid in active_effects
    )

    prompt = (
        "너는 판타지 세계관의 상태 정합성 감사관이다. 아래 엔티티에게 현재 해제되지 않은 "
        "상태(reversible status)가 걸려 있다.\n\n"
        f"엔티티: {entity_id}\n"
        f"현재 상태:\n{effect_lines}\n\n"
        f"새로 입력된 사건 문장: {raw_text}\n\n"
        "이 문장이 위 상태와 어떤 관계인지 다음 중 정확히 하나의 JSON으로 답하라:\n"
        '1) 자연스럽게 양립: {"result": "ok"}\n'
        '2) 위 상태를 해제하는 행동: {"result": "clears", "status_effect_id": "해당 상태 id", "reason": "판단 근거"}\n'
        '3) 위 상태와 상충될 가능성: {"result": "conflict", "status_effect_id": "해당 상태 id", "reason": "판단 근거"}\n'
        "JSON 이외의 텍스트는 출력하지 마라."
    )

    raw = _invoke_llm(prompt)
    data = _extract_json(raw)
    result = data.get("result")

    if result == "clears":
        return Judgment(
            type="clears_status",
            reason=data.get("reason", ""),
            entity_id=entity_id,
            status_effect_id=data.get("status_effect_id"),
        )
    if result == "conflict":
        return Judgment(
            type="conflict",
            reason=data.get("reason", ""),
            entity_id=entity_id,
            status_effect_id=data.get("status_effect_id"),
        )
    return None


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
    judgments = []

    hard_rule_docs = _get_hard_rule_texts()
    rule_judgment = check_rule_violation(entities, raw_text, hard_rule_docs)
    if rule_judgment is not None:
        judgments.append(rule_judgment)

    notes_judgment = check_notes_conflict(entities, raw_text)
    if notes_judgment is not None:
        judgments.append(notes_judgment)

    return judgments


def run_rag_checks(entities: list, raw_text: str, event_year: int) -> list:
    print(f"[rag_check] run_rag_checks 호출: entities={entities}, year={event_year}")
    judgments = []

    # check_rule_violation gets the canonical hard-rule texts plus each
    # involved entity's own stored context (Phase 10 patch 10, B) — not the
    # generic similarity-search context_docs, since mixing in unrelated
    # retrieved documents was observed to dilute the prompt enough that an
    # actual violation went undetected. retrieve_context() stays available
    # as a general-purpose utility, just not fed into this specific check.
    hard_rule_docs = _get_hard_rule_texts()
    rule_judgment = check_rule_violation(entities, raw_text, hard_rule_docs)
    if rule_judgment is not None:
        judgments.append(rule_judgment)

    notes_judgment = check_notes_conflict(entities, raw_text)
    if notes_judgment is not None:
        judgments.append(notes_judgment)

    for entity_id in entities:
        status_judgment = check_status_consistency(entity_id, raw_text, event_year)
        if status_judgment is not None:
            judgments.append(status_judgment)

    return judgments
