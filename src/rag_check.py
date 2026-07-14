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
# 2-2. World-rule violation
# ---------------------------------------------------------------------------

def check_rule_violation(raw_text: str, context_docs: list) -> Judgment | None:
    docs = "\n".join(f"- {d}" for d in context_docs) if context_docs else "(관련 규칙 없음)"
    prompt = (
        "너는 판타지 세계관의 규칙 감사관이다. 아래는 이 세계관에서 확정된 규칙 및 관련 문서다. "
        "새로 입력된 사건 문장이 이 규칙을 위반하는지 판단하라.\n\n"
        f"세계관 규칙/문서:\n{docs}\n\n"
        f"사건 문장: {raw_text}\n\n"
        "규칙이 특정 전제조건(도구, 자격, 재료 등)을 요구하는데, 사건 문장에 그 전제조건이 "
        "충족되었다는 언급이 전혀 없다면 — 굳이 결여를 명시하지 않았더라도 — 위반 가능성이 "
        "있는 것으로 판단하라. 규칙이 금지하는 행위 자체가 문장에 등장하면 전제조건 충족 여부를 "
        "확인할 길이 없는 이상 위반 쪽으로 판단하는 것이 기본값이다.\n\n"
        "규칙을 위반할 가능성이 있으면:\n"
        '{"violation": true, "reason": "위반이라고 판단한 근거", "confidence": 0.0에서 1.0 사이 숫자}\n'
        "위반 가능성이 없으면:\n"
        '{"violation": false}\n'
        "위 JSON 형식으로만 답하고 다른 텍스트는 출력하지 마라."
    )

    raw = _invoke_llm(prompt)
    data = _extract_json(raw)
    if not data.get("violation"):
        return None

    return Judgment(
        type="rule_violation",
        reason=data.get("reason", ""),
        confidence=data.get("confidence"),
    )


# ---------------------------------------------------------------------------
# 2-3. Notes-based qualitative conflict
# ---------------------------------------------------------------------------

def check_notes_conflict(entities: list, raw_text: str) -> Judgment | None:
    notes_lines = []
    for entity_id in entities:
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        record = storage.get_entity(category, entity_id)
        if not record:
            continue
        if record.get("notes"):
            notes_lines.append(f"{entity_id}: {record['notes']}")
        # A character's race carries its own notes (e.g. dietary restrictions)
        # that the character itself doesn't repeat.
        if category == "character" and record.get("race"):
            race_record = storage.get_entity("race", record["race"])
            if race_record and race_record.get("notes"):
                notes_lines.append(f"{record['race']}: {race_record['notes']}")

    if not notes_lines:
        return None

    notes_block = "\n".join(f"- {line}" for line in notes_lines)
    prompt = (
        "너는 판타지 세계관의 설정 감사관이다. 아래는 관련 엔티티들의 기존 설정(notes)이다. "
        "새로 입력된 사건 문장이 이 설정과 모순되는지 판단하라.\n\n"
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
        return None

    return Judgment(type="notes_conflict", reason=data.get("reason", ""))


# ---------------------------------------------------------------------------
# 2-4. Reversible status consistency
# ---------------------------------------------------------------------------

def check_status_consistency(entity_id: str, raw_text: str, event_year: int) -> Judgment | None:
    """Gated by event_year (Phase 9 status-range patch): if no status range
    on this entity actually covers event_year, there's nothing for the event
    to be consistent or inconsistent *with* at that point in time, so skip
    the LLM call entirely rather than asking it to judge against a status
    that (from the timeline's perspective) hadn't started yet, or had
    already ended, when this event happened."""
    active_effects = storage.get_active_statuses_at(entity_id, event_year)
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

def run_rag_checks(entities: list, raw_text: str, event_year: int) -> list:
    judgments = []

    # check_rule_violation gets ONLY the canonical hard-rule texts, not the
    # generic similarity-search context_docs — mixing in unrelated retrieved
    # documents was observed to dilute the prompt enough that an actual
    # violation went undetected. retrieve_context() stays available as a
    # general-purpose utility, just not fed into this specific check.
    hard_rule_docs = _get_hard_rule_texts()
    rule_judgment = check_rule_violation(raw_text, hard_rule_docs)
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
