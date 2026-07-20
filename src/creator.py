"""Creator (Phase 10 patch 22) — an autonomous reflection-loop feature that
drafts a multi-event narrative from a short user request, has Inspector
(the existing Step 4/5 validation pipeline) check each drafted event, and
retries on rejection until it passes or a retry cap is hit.

Unlike the normal chat pipeline, Creator never calls inference.infer_event —
its own narrative-composition logic (event count, point/duration, entity/
target, predicate) replaces Step 3's job entirely for this flow. Each
drafted event still gets real Korean prose (used as both the timeline
record's notes and the raw_text fed into rag_check's checks), so Step 4/5
run completely unmodified.
"""

import json
import re
from dataclasses import dataclass, field

from . import config, hard_check, rag_check, schema, storage

MAX_EVENTS = 5


@dataclass
class YearWindow:
    lower: int | None  # None = unbounded below
    upper: int | None  # None = unbounded above (still exists/ongoing)
    possible: bool
    reason: str | None = None  # set only when not possible
    per_entity: dict = field(default_factory=dict)  # entity_id -> (lower, upper)


def compute_year_window(entity_ids: list) -> YearWindow:
    """Intersects every entity's own existence range (hard_check.
    get_existence_range) into one window a Creator-generated story must fit
    within. `possible=False` means the entities' existence ranges never
    overlap at all (e.g. one died before another was born) — the caller
    should reject the request before ever invoking Creator, not burn a
    retry loop on something that can never pass Inspector."""
    per_entity = {}
    for entity_id in entity_ids:
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        per_entity[entity_id] = hard_check.get_existence_range(category, entity_id)

    lower = None
    upper = None
    for e_lower, e_upper in per_entity.values():
        if e_lower is not None:
            lower = e_lower if lower is None else max(lower, e_lower)
        if e_upper is not None:
            upper = e_upper if upper is None else min(upper, e_upper)

    if lower is not None and upper is not None and lower > upper:
        ended_before = [eid for eid, (_, u) in per_entity.items() if u is not None and u < lower]
        started_after = [eid for eid, (l, _) in per_entity.items() if l is not None and l > upper]
        reason = (
            f"{', '.join(ended_before)}의 존재가 끝난 시점({upper}년)이 "
            f"{', '.join(started_after)}의 존재가 시작된 시점({lower}년)보다 이릅니다 — "
            f"함께 존재하는 기간이 없습니다."
        )
        return YearWindow(lower=lower, upper=upper, possible=False, reason=reason, per_entity=per_entity)

    return YearWindow(lower=lower, upper=upper, possible=True, per_entity=per_entity)


# ---------------------------------------------------------------------------
# Narrative composition — Creator's own replacement for Step 3 in this flow
# ---------------------------------------------------------------------------

@dataclass
class DraftEvent:
    event_type: str  # "point" | "duration"
    notes: str
    involved_entities: list = field(default_factory=list)
    year: int | None = None
    start_year: int | None = None
    end_year: int | None = None
    duration_effect: dict | None = None


@dataclass
class NarrativeDraft:
    events: list  # list[DraftEvent]
    # Creator's own unconstrained judgment of how many events this story
    # ideally wants — reported honestly even when `events` itself had to
    # comply with a single-year constraint (verified: a 4-event story
    # compressed into one year still reports natural_event_count=4, not a
    # constraint-distorted 1), so the caller can detect a count/year-shape
    # mismatch (spec section B) purely as `is_single_year and
    # natural_event_count > 1` without a wasted extra LLM round-trip when
    # the user picks "compress" rather than "widen the range". An earlier
    # version also asked the LLM to self-report a `would_prefer_range`
    # boolean directly, but that came back inconsistent (False even for
    # the same 4-into-1-year case) — dropped in favor of this simpler,
    # structural signal that doesn't depend on a second subjective judgment.
    natural_event_count: int


def _get_llm():
    from langchain_openai import ChatOpenAI

    # temperature=0.7, unlike every other LLM call in this codebase
    # (always 0) — those are classification/judgment calls that need to be
    # reproducible; this one is creative drafting, and a retry after
    # Inspector rejection should actually explore a different narrative,
    # not deterministically regenerate the same rejected draft.
    return ChatOpenAI(model=config.get_model("reasoning"), temperature=0.7)


def _invoke_llm(prompt: str) -> str:
    response = _get_llm().invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾을 수 없습니다: {raw!r}")
    return json.loads(match.group(0))


def _entity_context_block(resolved_entities: dict) -> str:
    lines = []
    for tag, entity_id in resolved_entities.items():
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        record = storage.get_entity(category, entity_id)
        if not record:
            continue
        parts = [f"분류={category}"]
        summary = rag_check.entity_field_summary(record)
        if summary:
            parts.append(summary)
        if record.get("notes"):
            parts.append(f"notes={record['notes']}")
        lines.append(f'{entity_id} ("{tag}"): ' + ", ".join(parts))
    return "\n".join(f"- {line}" for line in lines) if lines else "(참고할 엔티티 정보 없음)"


def compose_narrative(
    resolved_entities: dict,
    request_text: str,
    lower: int,
    upper: int,
    feedback: str | None = None,
    supplement: str | None = None,
) -> NarrativeDraft:
    """Draft a multi-event narrative for `request_text`, entirely within
    [lower, upper] (inclusive, both concrete ints — the caller resolves any
    open-ended YearWindow into concrete bounds before calling this, e.g.
    via user confirmation). Bypasses inference.infer_event entirely — this
    function decides event_type/duration_effect/predicate/target itself,
    replacing Step 3's job for this flow (see module docstring); Step 4/5
    validation (Inspector) still runs against the notes text this produces,
    completely unmodified.

    `feedback`, when given, is Inspector's rejection reason(s) from a prior
    failed attempt in the same retry loop. `supplement` is an optional
    user-provided instruction from a [Redo] request — added on top of the
    original request, never replacing it."""
    entity_list = "\n".join(f'- "{tag}" -> {entity_id}' for tag, entity_id in resolved_entities.items())
    entity_context = _entity_context_block(resolved_entities)
    valid_ids = ", ".join(resolved_entities.values())

    all_status_effects = schema.load_status_effects()
    status_effect_options = "\n".join(
        f"- {s['id']} ({s['label']})" for s in all_status_effects if s.get("type", "individual") == "individual"
    ) or "(등록된 개인 상태 predicate 없음)"
    relational_predicate_options = "\n".join(
        f"- {s['id']} ({s['label']})" for s in all_status_effects if s.get("type") == "relational"
    ) or "(등록된 관계형 predicate 없음)"

    is_single_year = lower == upper
    year_constraint = (
        f"{lower}년 (단일 연도 — 모든 사건은 반드시 이 연도 하나로만 채워야 한다)"
        if is_single_year
        else f"{lower}년 ~ {upper}년 (이 범위 밖의 연도는 절대 쓰지 마라)"
    )

    single_year_instruction = (
        "\n\n이 요청은 단일 연도 하나로 제한되어 있다. natural_event_count는 이 제약과 "
        "무관하게, '만약 제약이 없었다면 몇 개의 사건으로 구성하는 게 이상적이었을지'를 "
        "정직하게 보고하라 — 실제로 events를 몇 개 작성했는지에 맞춰 축소해서 보고하지 마라 "
        "(예: 이상적으로는 4개가 자연스러운 이야기라면, 단일 연도 제약 때문에 실제로는 다르게 "
        "압축해서 작성하더라도 natural_event_count는 여전히 4여야 한다). events 자체는 "
        "그럼에도 불구하고 반드시 위 단일 연도 제약을 지켜서 최대한 압축된 형태로 작성하라."
        if is_single_year
        else ""
    )
    feedback_block = (
        f"\n\n이전 시도가 다음 이유로 반려되었다 — 이번에는 이 문제를 피해서 다시 구성하라:\n{feedback}"
        if feedback
        else ""
    )
    supplement_block = (
        f"\n\n사용자가 재생성 시 추가로 요청한 지침(원래 요청에 덧붙여 반영하라): {supplement}"
        if supplement
        else ""
    )

    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 이야기 기획자(Creator)다. 사용자의 짧은 요청을 "
        "받아, 그 요청을 표현하는 하나 이상의 사건 기록(timeline record) 초안을 스스로 작성하라. "
        "절대 새로운 엔티티를 지어내지 말고, 아래 확정된 엔티티만 사용하라.\n\n"
        f"확정된 엔티티:\n{entity_list}\n\n"
        f"엔티티 정보:\n{entity_context}\n\n"
        f"사용 가능한 entity_id: {valid_ids}\n\n"
        f"사용자 요청: {request_text}\n\n"
        f"허용된 연도: {year_constraint}\n\n"
        "=== 사건 개수/구성 판단 ===\n"
        "이 서사가 몇 개의 사건 기록으로 표현되는 게 자연스러운지 스스로 판단하라 — 고정된 "
        f"개수나 하한은 없다(응집도 높은 단일 사건이면 1개로 충분하다), 상한은 {MAX_EVENTS}개다. "
        "예: '원수가 됐다'는 보통 응집도 높은 단일 사건으로 충분하다. '사랑에 빠지기까지'처럼 "
        "과정 자체가 여러 단계(만남, 데이트, 고백 등)로 구성되는 게 자연스러운 서사는 여러 개의 "
        "point 사건으로 나누는 게 좋다.\n\n"
        "=== duration 이벤트 포함 여부 ===\n"
        "point 사건들의 결과가 실제로 지속되는 상태/관계의 성립으로 자연스럽게 귀결되는 경우에만 "
        "마지막에 duration 이벤트를 추가하라 — 여러 사건을 만든다고 항상 duration도 만들어야 "
        "하는 건 아니다. 예: 사랑 이야기라면 마지막에 연인 관계 duration을 추가하는 게 자연스럽지만, "
        "단순히 바보짓을 하는 이야기라면 point만으로 완결되고 duration은 불필요하다.\n\n"
        "duration_effect.predicate: 대상이 없는 개인 상태라면 아래 등록된 id 중 하나를 써라:\n"
        f"{status_effect_options}\n"
        "대상이 있는 관계라면, 이미 등록된 관계형 predicate 목록을 먼저 확인하고 상황에 맞는 게 "
        f"있으면 재사용하라:\n{relational_predicate_options}\n"
        "마땅히 재사용할 것이 없을 때만 새로운 predicate 이름을 자유롭게 만들어라 — 새 이름은 "
        "이후 별도 확인 절차를 거치므로 지어내는 것 자체는 괜찮다.\n\n"
        "각 point 사건에는 notes(실제 있었던 일을 서술하는 완결된 한국어 문장 — 이 문장은 이후 "
        "세계관 규칙/설정 모순 검증에 그대로 쓰이므로, 검증 가능하도록 구체적으로 서술하라)와 "
        "involved_entities(관련된 entity_id 목록)를 채워라. 각 duration 사건에는 notes와 "
        "duration_effect(entity, predicate, target, action='set', start_year)를 채워라."
        f"{single_year_instruction}{feedback_block}{supplement_block}\n\n"
        "아래 JSON 형식으로만 답하라 (다른 설명 금지):\n"
        "{\n"
        '  "natural_event_count": 정수 (이 서사에 이상적인 사건 개수, 제약 없이 판단),\n'
        '  "events": [\n'
        "    {\n"
        '      "event_type": "point 또는 duration",\n'
        '      "notes": "한국어 문장",\n'
        '      "involved_entities": ["entity_id", ...],\n'
        '      "year": "point일 때만, 정수 또는 null",\n'
        '      "start_year": "duration일 때만, 정수 또는 null",\n'
        '      "end_year": "duration이고 이미 종료된 경우만, 정수 또는 null",\n'
        '      "duration_effect": {"entity": "entity_id", "predicate": "...", '
        '"target": "entity_id 또는 null", "action": "set"} 또는 null\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    raw = _invoke_llm(prompt)
    data = _extract_json(raw)

    events = [
        DraftEvent(
            event_type=e.get("event_type", "point"),
            notes=e.get("notes", ""),
            involved_entities=e.get("involved_entities") or [],
            year=e.get("year"),
            start_year=e.get("start_year"),
            end_year=e.get("end_year"),
            duration_effect=e.get("duration_effect"),
        )
        for e in (data.get("events") or [])[:MAX_EVENTS]
    ]

    return NarrativeDraft(
        events=events,
        natural_event_count=data.get("natural_event_count") or len(events),
    )


# ---------------------------------------------------------------------------
# Inspector — reuses Step 4 (check_rule_and_notes) + Step 5 (hard_check)
# unmodified, walking the draft sequentially so a later event's check can
# see earlier events in the same draft as context, even though nothing is
# written to storage until the whole batch is approved and saved.
# ---------------------------------------------------------------------------

@dataclass
class InspectionResult:
    approved: bool
    reason: str | None = None  # combined human-readable feedback, for Creator's retry or the final rejection message
    failed_event_index: int | None = None


def _event_involved(event: DraftEvent, fallback: list) -> list:
    if event.event_type == "duration" and event.duration_effect:
        involved = [
            v for v in (event.duration_effect.get("entity"), event.duration_effect.get("target")) if v
        ]
        if involved:
            return involved
    return event.involved_entities or fallback


def inspect_draft(resolved_entities: dict, draft: NarrativeDraft) -> InspectionResult:
    """Stops at the first rejected event — Creator retries the whole batch
    (spec: whole-batch retry, not per-event patching), so nothing is gained
    by continuing to check events past the first failure."""
    entity_ids = list(resolved_entities.values())
    hard_rule_docs = rag_check._get_hard_rule_texts()
    approved_context_lines = []  # this draft's own already-approved events' notes
    approved_years = {}  # entity_id -> [year, ...] already used earlier in this draft

    for i, event in enumerate(draft.events):
        involved = _event_involved(event, entity_ids)
        event_year = event.year if event.event_type == "point" else event.start_year
        candidate_years = [y for y in (event.year, event.start_year, event.end_year) if y is not None]

        for entity_id in involved:
            category = schema.category_from_id(entity_id)
            if category is None:
                continue
            extra_years = candidate_years + approved_years.get(entity_id, [])
            conflicts = hard_check.run_hard_checks(category, entity_id, extra_years=extra_years)
            blocking = [c for c in conflicts if c.severity == "blocking"]
            if blocking:
                reason = (
                    f"{i + 1}번째 사건(\"{event.notes}\")이 하드체크에 위반됩니다: "
                    + "; ".join(c.reason for c in blocking)
                )
                return InspectionResult(approved=False, reason=reason, failed_event_index=i)

        judgments = rag_check.check_rule_and_notes(
            involved, event.notes, hard_rule_docs, event_year, extra_context=approved_context_lines
        )
        if judgments:
            reasons = "; ".join(f"[{j.type}] {j.reason}" for j in judgments)
            reason = f"{i + 1}번째 사건(\"{event.notes}\")이 검증에 실패했습니다: {reasons}"
            return InspectionResult(approved=False, reason=reason, failed_event_index=i)

        for entity_id in involved:
            approved_context_lines.append(
                f"{entity_id}의 관련 기록(이번 초안 {i + 1}번째 사건): {event.notes}"
            )
            if candidate_years:
                approved_years.setdefault(entity_id, []).extend(candidate_years)

    return InspectionResult(approved=True)
