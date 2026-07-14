"""Event inference (Step 3) — Phase 10 rewrite.

Replaces Phase 3's infer_relationship_and_event: instead of an event summary
plus a list of separate relationship rows (which in practice produced ten
near-duplicate relationships per input), one input now produces exactly one
timeline record — either a "point" event (something that happened) or a
"duration" event (a status/relationship starting or ending). The LLM decides
which, and for duration events, whether this is a fresh start, an end, or
(given two years) a already-closed span.

entity_presence (Phase 9 patch D) is preserved and extended: an entity can
be *involved* in a point event (worth a pointer in its event history)
without being *present* at it (worth hard-checking this event's year
against its lifespan) — digging up someone's grave involves them without
implying they were alive for the digging.
"""

import json
import re
from dataclasses import dataclass, field

from . import config, schema


@dataclass
class InferredEvent:
    event_type: str  # "point" | "duration" — meaningless if is_single_event is False
    event_summary: str | None = None
    involved_entities: list = field(default_factory=list)
    duration_effect: dict | None = None
    entity_presence: dict = field(default_factory=dict)
    is_single_event: bool = True
    ambiguity_reason: str | None = None


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


def infer_event(resolved_entities: dict, raw_text: str, years: list) -> InferredEvent:
    entity_list = "\n".join(
        f'- "{tag}" -> {entity_id}' for tag, entity_id in resolved_entities.items()
    )
    valid_ids = ", ".join(resolved_entities.values())
    status_effect_options = "\n".join(
        f"- {s['id']} ({s['label']})" for s in schema.load_status_effects()
    )
    years_text = ", ".join(str(y) for y in years)

    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 사건 기록자다.\n"
        "아래 엔티티들은 이미 확정되었다. 절대 새로운 엔티티를 지어내지 말고, "
        "duration_effect의 entity/target과 involved_entities는 반드시 아래 목록의 "
        "entity_id만 사용하라.\n\n"
        f"확정된 엔티티:\n{entity_list}\n\n"
        f"사용 가능한 entity_id: {valid_ids}\n\n"
        f"원문: {raw_text}\n"
        f"문장에서 추출된 연도(들): {years_text}\n\n"
        "먼저 이 문장이 하나의 시점/흐름에서 벌어진 사건인지 판단하라. 기준은 오직 "
        "'하나의 연도(시점)로 묶이는가'이지, 몇 명이 등장하거나 몇 개의 동작이 나열되는가가 "
        "아니다. 한 장면 안에서 여러 인물이 서로 얽혀 벌어지는 일련의 행동은 — 예를 들어 "
        "'A가 B와 있다가 C에게 들켜 두들겨 맞고 쫓겨났고, B는 그걸 보고 웃은 뒤 C와 함께 "
        "갔다' — 등장인물과 동작이 여러 개라도 전부 같은 시점의 한 장면이므로 정상적인 "
        "단일 사건(point)이다. involved_entities에 관련된 entity_id를 전부 담으면 그걸로 "
        "충분하다 — 각 인물의 행동을 억지로 분리해서 별개 사건으로 취급하지 마라. "
        "is_single_event를 false로 판단해야 하는 경우는 오직, 서로 다른 연도(시점)에 벌어진 "
        "서로 무관한 별개의 사건들이 한 문장에 함께 나열된 경우뿐이다(예: '쟝이 2080년에 "
        "술을 마셨고, 랄프가 2200년에 죽었다' — 완전히 다른 시점, 다른 맥락의 두 사건). "
        "같은 시점의 상태 시작/끝을 함께 서술하는 경우(예: '2000년부터 2010년까지')도 "
        "정상적인 단일 사건이다. 이 두 예외에 해당하지 않는 한 is_single_event는 항상 "
        "true여야 한다 — 애매하면 false가 아니라 true를 기본값으로 하라. false일 때만 "
        "ambiguity_reason에 이유를 설명하고, 그 경우 다른 필드는 채우지 않아도 된다.\n\n"
        "is_single_event가 true라면, 이 사건이 다음 중 무엇인지 판단하라:\n\n"
        "1) point (점 이벤트) — 특정 시점에 벌어진 일회성 사건. 지속되는 상태나 관계의 "
        "시작/종료를 암시하지 않는다.\n"
        "2) duration (기간 이벤트) — 어떤 엔티티의 지속적인 상태(수감, 봉인 등)나 다른 "
        "엔티티와의 관계(소속, 적대 등)가 시작되거나 끝나는 서술.\n\n"
        "중요: 문장이 특정 행동을 묘사하더라도, 그 행동이 상태의 시작이나 종료를 암시한다면 "
        "point가 아니라 duration으로 판단하라 — 표면적인 동사(무엇을 했는가)가 아니라 그 "
        "행동이 초래하는 상태 변화가 기준이다. 예: '탈출했다/풀려났다/석방됐다'는 수감 "
        "상태의 종료(duration, clear, predicate=imprisoned), '붙잡혔다/투옥됐다/감금됐다'는 "
        "수감 상태의 시작(duration, set, predicate=imprisoned), '봉인에서 풀려났다'는 봉인 "
        "해제(duration, clear, predicate=sealed), '실종됐다/사라졌다'는 실종 시작(duration, "
        "set, predicate=missing), '저주에 걸렸다'는 저주 시작(duration, set, "
        "predicate=cursed), '크게 다쳐 쓰러졌다'는 부상/중태 시작(duration, set, "
        "predicate=incapacitated), '의식을 되찾았다/회복했다'는 부상/중태 종료(duration, "
        "clear, predicate=incapacitated)다. 이런 암시가 전혀 없는 사건(전투, 대화, 이동, "
        "물건 획득 등 그 자체로 끝나는 행동)만 point로 판단하라. 주의: '나타났다/돌아왔다/"
        "보였다'처럼 단순히 등장하거나 눈에 띄었다는 서술만으로는 실종(missing) 해제로 "
        "판단하지 마라 — 그 인물이 이전에 실종/행방불명 상태였다는 점이 문장이나 이미 알려진 "
        "맥락에 명시적으로 있을 때만 duration/clear/missing으로 판단하고, 그런 근거가 없으면 "
        "그냥 point로 판단하라. 이 신중함은 다른 상태(수감, 봉인, 저주, 부상/중태)에도 "
        "동일하게 적용된다 — 상태 변화 동사가 명확할 때만 duration으로 판단하고, 조금이라도 "
        "불확실하면 point를 기본값으로 하라.\n\n"
        "duration이면 action을 다음 중 하나로 판단하라:\n"
        "  - set: 연도가 하나이고, 새로운 상태/관계가 이 연도에 시작됨 (end_year는 없음)\n"
        "  - clear: 연도가 하나이고, 기존 상태/관계가 이 연도에 끝남 (예: '풀려났다', "
        "'해체됐다')\n"
        "  - set_closed: 연도가 둘이고, 같은 상태/관계의 시작과 끝을 모두 서술함 "
        "(start_year=작은 연도, end_year=큰 연도)\n\n"
        "duration_effect.predicate: 대상이 없는 개인 상태(수감, 봉인 등)라면 반드시 아래 "
        f"목록의 id 중 하나를 써야 하며 target은 null로 둔다:\n{status_effect_options}\n"
        "대상이 있는 관계(소속, 적대, 아는 사이 등)라면 predicate는 자유 텍스트로 쓰고 "
        "target에 상대 entity_id를 채워라.\n\n"
        "point면 event_summary(한 줄 요약)와 involved_entities(이 사건에 실제로 관련된 "
        "entity_id 목록 — 보통 사용 가능한 entity_id 전부)를 채워라.\n\n"
        "entity_presence: 사용 가능한 entity_id 각각에 대해, 이 사건이 그 엔티티가 이 "
        "문장의 연도(들)에 실제로 살아있거나 존재/활동하고 있었음을 전제로 하는지(true) "
        "아니면 이미 지나간 사실이나 유산·기록물에 대한 참조일 뿐인지(false) 판단하라. "
        "사건을 직접 행하는 주체는 거의 항상 true다. 예: '[밥]이 [쟝]의 무덤을 파헤쳤다'에서 "
        "밥은 true, 쟝은 false다(쟝은 이미 죽었을 수 있는 과거의 흔적일 뿐 이 사건에 살아서 "
        "참여하지 않는다 — 다만 이 사건은 쟝과 관련은 있으므로 involved_entities에는 "
        "포함시켜라). '[데이비드]가 [쟝]과 놀았다'에서는 데이비드와 쟝 모두 true다.\n\n"
        "아래 JSON 형식으로만 답하라 (다른 설명 금지):\n"
        "{\n"
        '  "is_single_event": true 또는 false,\n'
        '  "ambiguity_reason": "is_single_event가 false일 때만 이유 설명" 또는 null,\n'
        '  "event_type": "point" 또는 "duration",\n'
        '  "event_summary": "point일 때만 채움, 사건 한 줄 요약" 또는 null,\n'
        '  "involved_entities": ["point일 때만 채움, entity_id 목록"],\n'
        '  "duration_effect": {\n'
        '    "entity": "entity_id", "predicate": "...", "target": "entity_id 또는 null",\n'
        '    "action": "set 또는 clear 또는 set_closed",\n'
        '    "start_year": 정수 또는 null, "end_year": 정수 또는 null\n'
        '  } 또는 null,\n'
        '  "entity_presence": {"entity_id": true 또는 false, ...}\n'
        "}\n"
    )

    last_error = None
    for _ in range(2):
        raw = _invoke_llm(prompt)
        try:
            data = _extract_json(raw)
            if not data.get("is_single_event", True):
                return InferredEvent(
                    event_type="point",
                    is_single_event=False,
                    ambiguity_reason=(
                        data.get("ambiguity_reason")
                        or "여러 사건 또는 대상이 한 문장에 섞여 있어 하나의 기록으로 만들 수 없습니다."
                    ),
                )
            return InferredEvent(
                event_type=data["event_type"],
                event_summary=data.get("event_summary"),
                involved_entities=data.get("involved_entities") or [],
                duration_effect=data.get("duration_effect"),
                entity_presence=data.get("entity_presence") or {},
                is_single_event=True,
                ambiguity_reason=None,
            )
        except (ValueError, KeyError) as exc:
            last_error = exc

    raise ValueError(f"사건 추론 결과를 파싱하지 못했습니다: {last_error}")
