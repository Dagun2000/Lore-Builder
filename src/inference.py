"""Relationship/event inference (Step 3) — Phase 3.

The only LLM call here reasons about *what happened* between entities that
are already resolved (anchored) — it must never invent new entities. Uses
the reasoning-tier model (config.get_model("reasoning")) since this needs
nuanced judgment, unlike Phase 2's plain classification.
"""

import json
import re
from dataclasses import dataclass, field

from . import config, schema


@dataclass
class InferredEvent:
    event_summary: str
    relationships: list = field(default_factory=list)
    status_effect: dict | None = None
    entity_presence: dict = field(default_factory=dict)


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


def infer_relationship_and_event(
    resolved_entities: dict, raw_text: str, year: int
) -> InferredEvent:
    entity_list = "\n".join(
        f'- "{tag}" -> {entity_id}' for tag, entity_id in resolved_entities.items()
    )
    valid_ids = ", ".join(resolved_entities.values())
    status_effect_options = "\n".join(
        f"- {s['id']} ({s['label']})" for s in schema.load_status_effects()
    )

    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 사건 기록자다.\n"
        "아래 엔티티들은 이미 확정되었다. 절대 새로운 엔티티를 지어내지 말고, "
        "relationships의 subject/object는 반드시 아래 목록의 entity_id만 사용하라.\n\n"
        f"확정된 엔티티:\n{entity_list}\n\n"
        f"사용 가능한 entity_id: {valid_ids}\n\n"
        f"원문: {raw_text}\n"
        f"연도: {year}\n\n"
        "이 문장에서 벌어진 일을 분석해 아래 JSON 형식으로만 답하라 (다른 설명 금지):\n"
        "{\n"
        '  "event_summary": "사건을 한 줄로 요약한 문장",\n'
        '  "relationships": [{"subject": "entity_id", "predicate": "관계/행동 서술어", "object": "entity_id"}],\n'
        '  "status_effect": {"entity": "entity_id", "effect": "status_effect id", "action": "set 또는 clear"} 또는 null,\n'
        '  "entity_presence": {"entity_id": true 또는 false, ...}\n'
        "}\n\n"
        f"status_effect.effect는 반드시 아래 목록의 id 중 하나여야 하며, 목록에 없는 값은 절대 "
        f"지어내지 마라:\n{status_effect_options}\n\n"
        "단순한 몸싸움이나 가벼운 사건만으로는 status_effect를 채우지 마라. 위 목록의 상태 중 "
        "하나가 명확하고 결정적으로 새로 부여되거나 해제되는 경우에만 채우고, 조금이라도 "
        "애매하면 null로 두어라.\n\n"
        "entity_presence: '사용 가능한 entity_id' 각각에 대해, 이 사건이 그 엔티티가 이 연도에 "
        "실제로 살아있거나 존재/활동하고 있었음을 전제로 하는지(true) 아니면 이미 지나간 사실이나 "
        "유산·기록물에 대한 참조일 뿐인지(false) 판단하라. 사건을 직접 행하는 주체는 거의 항상 "
        "true다. 반대로 무덤, 유물, 과거 기록처럼 그 엔티티의 흔적만 다뤄지고 본인은 이 사건에 "
        "실제로 등장/참여하지 않는 경우는 false다. 예: '[밥]이 [쟝]의 무덤을 파헤쳤다'에서 밥은 "
        "true, 쟝은 false (쟝은 이미 죽었을 수 있는 과거의 흔적일 뿐 이 사건에 살아서 참여하지 "
        "않음). '[데이비드]가 [쟝]과 놀았다'에서는 데이비드와 쟝 모두 true (둘 다 이 시점에 함께 "
        "존재하며 상호작용함). 목록의 모든 entity_id에 대해 빠짐없이 판단하라."
    )

    last_error = None
    for _ in range(2):
        raw = _invoke_llm(prompt)
        try:
            data = _extract_json(raw)
            return InferredEvent(
                event_summary=data["event_summary"],
                relationships=data.get("relationships") or [],
                status_effect=data.get("status_effect"),
                entity_presence=data.get("entity_presence") or {},
            )
        except (ValueError, KeyError) as exc:
            last_error = exc

    raise ValueError(f"관계/사건 추론 결과를 파싱하지 못했습니다: {last_error}")
