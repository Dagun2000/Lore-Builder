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

from . import config, schema, storage


@dataclass
class InferredEvent:
    event_type: str  # "point" | "duration" — meaningless if is_single_event is False
    event_summary: str | None = None
    involved_entities: list = field(default_factory=list)
    duration_effect: dict | None = None
    entity_presence: dict = field(default_factory=dict)
    # Phase 10 patch 6 (B): entity_id -> True if this event explicitly ends
    # that entity permanently (death/destruction/disbandment). Lets the
    # caller offer to update an *existing* entity's terminal field
    # (death_year, disbanded_year, destroyed_year) as a narrow, explicit
    # exception to "chat never edits an existing entity's fields" — a brand
    # new entity handles this at creation time instead (see
    # inference.infer_new_entity_attributes), so this only matters for
    # entities that already existed before this input.
    terminal_entities: dict = field(default_factory=dict)
    # Mirrors terminal_entities but for the opposite boundary: entity_id ->
    # True if this event explicitly states that *already-existing* entity's
    # own origin (birth/founding/creation). A brand-new entity gets its
    # lifecycle_start field filled at creation time instead (see
    # infer_new_entity_attributes) — this only matters for an entity that
    # already had other events on record before this input, so hard_check
    # can catch "born after already having done something" the same
    # deterministic way it catches "died before doing something later".
    genesis_entities: dict = field(default_factory=dict)
    # Phase 10 patch 6.5 (C): a single cohesive scene (is_single_event True)
    # can still need more than one timeline record — e.g. two different
    # entities' membership facts plus a shared duel/death. Each item here
    # has the same shape as this dataclass's own event_type/event_summary/
    # involved_entities/duration_effect quartet; archivist.build_diff
    # processes [self] + additional_records uniformly, no record-count cap.
    # Left empty for the (overwhelmingly common) single-record case.
    additional_records: list = field(default_factory=list)
    is_single_event: bool = True
    ambiguity_reason: str | None = None


def _get_llm(tier: str = "reasoning"):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.get_model(tier), temperature=0)


def _invoke_llm(prompt: str, tier: str = "reasoning") -> str:
    response = _get_llm(tier).invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾을 수 없습니다: {raw!r}")
    return json.loads(match.group(0))


_ENTITY_IDENTITY_EXCLUDED = {"id", "name", "event_ids", "lifespan_check_ack"}


def _entity_identity_lines(resolved_entities: dict) -> list:
    """Each resolved entity's own stored category + fields + notes (Phase 10
    patch 13, A) — Step 3 used to judge event_type/predicate purely from the
    raw sentence, with zero visibility into what a destination/object
    actually IS (e.g. a location whose notes say "세계에서 가장 거대한
    감옥이다"), so a neutral verb like "끌려갔다" never registered as an
    imprisonment no matter where the entity was taken. Deliberately a small,
    local helper rather than reusing rag_check.entity_field_summary — Step 3
    and Step 4 stay independent modules even though the flattening logic is
    similar; this is an event-type/predicate judgment, not a contradiction
    check."""
    lines = []
    for tag, entity_id in resolved_entities.items():
        category = schema.category_from_id(entity_id)
        if category is None:
            continue
        record = storage.get_entity(category, entity_id)
        if not record:
            continue
        parts = [f"분류={category}"] + [
            f"{name}={value}"
            for name, value in record.items()
            if name not in _ENTITY_IDENTITY_EXCLUDED and value not in (None, "", [])
        ]
        lines.append(f"{entity_id} (\"{tag}\"): " + ", ".join(parts))
    return lines


def infer_event(resolved_entities: dict, raw_text: str, years: list) -> InferredEvent:
    entity_list = "\n".join(
        f'- "{tag}" -> {entity_id}' for tag, entity_id in resolved_entities.items()
    )
    identity_lines = _entity_identity_lines(resolved_entities)
    identity_block = (
        "\n".join(f"- {line}" for line in identity_lines)
        if identity_lines else "(참고할 엔티티 정보 없음)"
    )
    valid_ids = ", ".join(resolved_entities.values())
    all_status_effects = schema.load_status_effects()

    def _effect_line(s: dict) -> str:
        line = f"- {s['id']} ({s['label']})"
        if s.get("notes"):
            line += f": {s['notes']}"
        return line

    status_effect_options = "\n".join(
        _effect_line(s) for s in all_status_effects if s.get("type", "individual") == "individual"
    )
    relational_predicate_options = "\n".join(
        _effect_line(s) for s in all_status_effects if s.get("type") == "relational"
    ) or "(등록된 관계형 predicate 없음)"
    years_text = ", ".join(str(y) for y in years)

    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 사건 기록자다.\n"
        "아래 엔티티들은 이미 확정되었다. 절대 새로운 엔티티를 지어내지 말고, "
        "duration_effect의 entity/target과 involved_entities는 반드시 아래 목록의 "
        "entity_id만 사용하라.\n\n"
        f"확정된 엔티티:\n{entity_list}\n\n"
        f"엔티티 정체성 정보(이미 저장된 분류/필드/notes):\n{identity_block}\n\n"
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
        "정상적인 단일 사건이다. 이 예외에 해당하지 않는 한 is_single_event는 항상 "
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
        "clear, predicate=incapacitated), 물건을 '잃어버렸다/도난당했다'는 분실 시작"
        "(duration, set, predicate=lost), 그 물건을 '다시 찾았다/발견했다'는 분실 종료"
        "(duration, clear, predicate=lost)다 — 파괴/소실(destroyed)과는 다르다, 파괴는 "
        "되돌릴 수 없는 종료이므로 duration이 아니라 해당 엔티티 자신의 lifecycle 필드로 "
        "처리된다(이 함수의 범위 밖). 이런 암시가 전혀 없는 사건(전투, 대화, 이동, "
        "물건 획득 등 그 자체로 끝나는 행동)만 point로 판단하라. 주의: '나타났다/돌아왔다/"
        "보였다'처럼 단순히 등장하거나 눈에 띄었다는 서술만으로는 실종(missing) 해제로 "
        "판단하지 마라 — 그 인물이 이전에 실종/행방불명 상태였다는 점이 문장이나 이미 알려진 "
        "맥락에 명시적으로 있을 때만 duration/clear/missing으로 판단하고, 그런 근거가 없으면 "
        "그냥 point로 판단하라. 이 신중함은 다른 상태(수감, 봉인, 저주, 부상/중태)에도 "
        "동일하게 적용된다 — 상태 변화 동사가 명확할 때만 duration으로 판단하고, 조금이라도 "
        "불확실하면 point를 기본값으로 하라.\n\n"
        "단, 위의 '동사가 명확할 때만'이라는 기준에는 중요한 예외가 있다: 사건에 언급된 "
        "장소/사물이 이미 저장된 카테고리나 notes를 가지고 있다면(위 '엔티티 정체성 정보' "
        "참고), 그 정체성이 이번 행동의 성격을 암시하는지 확인하라. 예를 들어 목적지가 "
        "\"감옥\"으로 저장되어 있다면, \"끌려갔다\"/\"보내졌다\"/\"이송되었다\"처럼 투옥을 "
        "직접 명시하지 않는 중립적인 동사라도, 그 목적지로 이동시켰다는 서술 자체가 실질적으로는 "
        "수감(duration, set, predicate=imprisoned) 상태 변화를 의미한다. 동사 자체가 "
        "명시적인지 여부보다 행선지/대상의 정체성이 실질적으로 무엇을 의미하는지를 우선하라 — "
        "단 그 장소/사물에 그런 정체성을 암시하는 저장된 정보가 전혀 없다면, 중립적인 동사만으로 "
        "duration을 추정하지 말고 그냥 point로 판단하라.\n\n"
        "또 다른 독립적인 트리거(패치13의 목적지 정체성 판단과는 별개): 문장에 \"~로서\", "
        "\"~직위로\", \"~역할을 맡아\" 같은 역할/직위 부여 표현이 있다면, 목적지나 상황의 "
        "정체성과 무관하게 이것도 duration 이벤트로 판단하라(predicate는 자유 텍스트로, 예: "
        "employed_at, serves_as, target은 그 역할이 소속된 장소/조직의 entity_id). 같은 "
        "문장에 다른 구체적 사건(예: 폭행, 대화)이 함께 서술되어 있다면 그 사건도 놓치지 말고 "
        "채워라 — 역할 duration 사실과 그 사건이 하나의 event_type/duration_effect만으로는 "
        "다 담기지 않으면, 아래 additional_records를 통해 응집된 한 장면 안에서 duration과 "
        "point 둘 다 생성하라.\n\n"
        "duration이면 action을 다음 중 하나로 판단하라:\n"
        "  - set: 연도가 하나이고, 새로운 상태/관계가 이 연도에 시작됨 (end_year는 없음)\n"
        "  - clear: 연도가 하나이고, 기존 상태/관계가 이 연도에 끝남 (예: '풀려났다', "
        "'해체됐다')\n"
        "  - set_closed: 연도가 둘이고, 같은 상태/관계의 시작과 끝을 모두 서술함 "
        "(start_year=작은 연도, end_year=큰 연도)\n\n"
        "duration_effect.predicate: 대상이 없어도 되는 개인 상태(수감, 봉인 등)라면 아래 "
        f"목록의 id 중 하나를 써야 한다:\n{status_effect_options}\n"
        "대상이 있는 관계(소속, 적대, 아는 사이 등)라면, 이미 등록된 관계형 predicate 목록을 "
        f"먼저 확인하고 상황에 맞는 것이 있으면 재사용하라:\n{relational_predicate_options}\n"
        "마땅히 재사용할 것이 없을 때만 새로운 predicate 이름을 자유롭게 만들어라 — 새 이름은 "
        "이후 별도 확인 절차를 거쳐 목록에 등록되므로, 지어내는 것 자체는 괜찮다.\n\n"
        "duration_effect.target: predicate의 성격과 무관하게 target을 항상 채울 수 있는지 "
        "먼저 시도하라. 문장에 이 상태/사건의 대상(누가/어디서/무엇에 의해 등)이 명시되어 "
        "있고 그 대상이 이미 확정된 엔티티라면 반드시 target에 그 entity_id를 채운다 — "
        "\"수감/봉인처럼 흔히 대상이 없는 개인 상태\"라는 predicate의 일반적인 경향을 이유로 "
        "target 채우기 자체를 생략하지 마라. 예: '[쟝]이 [알카미아]에 수감되었다'는 "
        "predicate=imprisoned이면서 동시에 target=알카미아의 entity_id여야 한다(수감시킨 "
        "장소가 문장에 명시되어 있으므로). 문장에 대상이 전혀 언급되지 않은 경우(예: '저주받았다'"
        ")에만 target을 null로 둬라 — 명시되지 않은 대상을 추측해서 채우지는 마라.\n\n"
        "duration_effect의 target은 항상 entity_id 하나다 — 리스트를 넣지 마라. 그런데 "
        "문장이 셋 이상의 엔티티가 서로 동등하게 얽히는 상호적/그룹 관계(예: 'A, B, C가 모두 "
        "친구가 되었다', 'A, B, C가 동맹을 맺었다')를 서술한다면, 그 관계는 한 쌍(entity, "
        "target)짜리 레코드 하나로는 다 담을 수 없다 — 관련된 모든 두 엔티티 쌍마다 하나씩 "
        "레코드를 만들어라(예: A-B, A-C, B-C 세 개). 가장 핵심적인 쌍 하나만 위의 "
        "event_type/duration_effect에 채우고, 나머지 쌍들은 아래 additional_records에 "
        "각각 하나씩 채워라 — 특정 엔티티가 대상 목록에서 조용히 빠지는 일이 없어야 한다.\n\n"
        "point면 event_summary(한 줄 요약)와 involved_entities(이 사건에 실제로 관련된 "
        "entity_id 목록 — 보통 사용 가능한 entity_id 전부)를 채워라.\n\n"
        "entity_presence: 사용 가능한 entity_id 각각에 대해, 이 사건이 그 엔티티가 이 "
        "문장의 연도(들)에 실제로 살아있거나 존재/활동하고 있었음을 전제로 하는지(true) "
        "아니면 이미 지나간 사실이나 유산·기록물에 대한 참조일 뿐인지(false) 판단하라. "
        "사건을 직접 행하는 주체는 거의 항상 true다. 예: '[밥]이 [쟝]의 무덤을 파헤쳤다'에서 "
        "밥은 true, 쟝은 false다(쟝은 이미 죽었을 수 있는 과거의 흔적일 뿐 이 사건에 살아서 "
        "참여하지 않는다 — 다만 이 사건은 쟝과 관련은 있으므로 involved_entities에는 "
        "포함시켜라). '[데이비드]가 [쟝]과 놀았다'에서는 데이비드와 쟝 모두 true다.\n\n"
        "terminal_entities: 사용 가능한 entity_id 각각에 대해, 이 사건이 그 엔티티의 "
        "완전하고 돌이킬 수 없는 종료(죽음, 파괴, 해체 등)를 명시적으로 서술하는지(true) "
        "판단하라. 단순히 다치거나, 위험에 처하거나, 사라졌다(실종)는 정도로는 true가 아니다 "
        "— 죽음/파괴/해체처럼 명백히 최종적인 서술일 때만 true, 그 외 전부 false.\n\n"
        "genesis_entities: 사용 가능한 entity_id 각각에 대해, 이 사건이 그 엔티티 자신의 "
        "탄생/창단/창조(태어났다, 창단되었다, 만들어졌다 등)를 명시적으로 서술하는지(true) "
        "판단하라. terminal_entities와 정반대 방향의 판단이다 — 그 외 전부 false.\n\n"
        "additional_records (하나의 응집된 장면에 레코드가 여러 개 필요한 경우): "
        "is_single_event가 true인 하나의 응집된 장면이라도, 서로 다른 대상에 대한 여러 "
        "지속 관계/상태(duration) 사실이 함께 서술되어 있거나, 지속 관계/상태 서술과 별개의 "
        "point 사건이 같은 장면 안에 함께 있어서, 위의 단일 event_type/duration_effect 필드 "
        "하나로는 다 담을 수 없는 경우가 있다 — 예: '고트프리는 철왕국 소속이고, 레오파트는 "
        "아마조네스 소속이며, 둘은 결투해서 죽었다'(같은 장면 안의 소속 관계 2건 + 결투 사건 "
        "1건). 이런 경우, 가장 핵심적인 사실 하나만 위의 event_type/event_summary/"
        "involved_entities/duration_effect에 채우고, 나머지 사실들은 additional_records "
        "배열에 각각 하나의 레코드로 채워라 — 각 항목은 위와 동일한 형태({event_type, "
        "event_summary, involved_entities, duration_effect})를 가진다. 정보를 조용히 "
        "버리지 마라 — 장면에 있는 모든 duration/point 사실은 주 레코드나 additional_records "
        "중 어딘가에 반드시 포함되어야 한다. 추가로 담을 사실이 없으면 빈 배열로 둬라.\n\n"
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
        '  "entity_presence": {"entity_id": true 또는 false, ...},\n'
        '  "terminal_entities": {"entity_id": true 또는 false, ...},\n'
        '  "genesis_entities": {"entity_id": true 또는 false, ...},\n'
        '  "additional_records": [\n'
        '    {\n'
        '      "event_type": "point 또는 duration",\n'
        '      "event_summary": "point일 때만" 또는 null,\n'
        '      "involved_entities": ["point일 때만"],\n'
        '      "duration_effect": { ... 위와 동일한 형태 } 또는 null\n'
        "    }\n"
        "  ]\n"
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
            additional_records = [
                InferredEvent(
                    event_type=r["event_type"],
                    event_summary=r.get("event_summary"),
                    involved_entities=r.get("involved_entities") or [],
                    duration_effect=r.get("duration_effect"),
                )
                for r in (data.get("additional_records") or [])
                if r.get("event_type") in ("point", "duration")
            ]
            return InferredEvent(
                event_type=data["event_type"],
                event_summary=data.get("event_summary"),
                involved_entities=data.get("involved_entities") or [],
                duration_effect=data.get("duration_effect"),
                entity_presence=data.get("entity_presence") or {},
                terminal_entities=data.get("terminal_entities") or {},
                genesis_entities=data.get("genesis_entities") or {},
                additional_records=additional_records,
                is_single_event=True,
                ambiguity_reason=None,
            )
        except (ValueError, KeyError) as exc:
            last_error = exc

    raise ValueError(f"사건 추론 결과를 파싱하지 못했습니다: {last_error}")


# ---------------------------------------------------------------------------
# New-entity attribute extraction (Phase 10 patch 2, section A)
# ---------------------------------------------------------------------------

_ATTRIBUTE_FIELD_ROLES = ("lifecycle_start", "lifecycle_end")


def _attribute_candidate_fields(category: str) -> list:
    """Fields eligible for direct auto-fill on a brand-new entity: lifecycle
    year markers (role lifecycle_start/end, e.g. founded_year, death_year)
    plus any other *optional* enum field (e.g. character.gender). Required
    fields are always forced through the normal field-collection flow
    instead (never silently auto-filled), and reference/text/list fields are
    excluded — "explicitly stated" is only reliably checkable for a plain
    year or a closed set of enum options."""
    fields = []
    for f in schema.get_fields(category):
        if f.get("role") in _ATTRIBUTE_FIELD_ROLES:
            fields.append(f)
        elif f["type"] == "enum" and not f.get("required"):
            fields.append(f)
    return fields


def infer_new_entity_attributes(category: str, tag: str, context_sentence: str, years: list) -> dict:
    """For a brand-new entity only (never called on an existing one — see
    pipeline_session._resolve_entity_gen), split the sentence's content about
    this entity into three non-overlapping paths (Phase 10 patch 3, E):

      1. lifecycle fields — a year explicitly tied to a role
         lifecycle_start/end field (founded_year, death_year, ...) or an
         explicit optional-enum value (gender, ...) -> straight onto the
         entity's own fields, never an event by itself.
      2. a time-bound occurrence/status (something that *happened* or
         *started/ended* at a point in time) -> left untouched here; Step 3
         (infer_event) + archivist still turn that into a timeline record,
         exactly as before this patch.
      3. a persistent trait/rule/characteristic that is always true rather
         than something that happened ("여성만 가입 가능하다") -> the
         entity's own `notes` field, verbatim-ish, no event, no year
         required at all.

    Never infers or back-calculates a value for path 1 — only reports what's
    explicitly written — and only ever reports a year that's in `years`
    (this input's own extracted year list, which may be empty: a pure
    introduction sentence with no year is valid input).

    Phase 10 patch 4 (K): a lifecycle year is not always a *bare* fact — "100년에
    태어난... 110년에 술을 먹고 싸우다가 죽었다" ties death_year to a whole
    narrated scene, not just the number 110. Blindly dropping every
    consumed year from the caller's event-judgment pool (as patch 2/3 did)
    silently threw that scene away whenever it happened to land on the same
    year as the lifecycle fact. So each consumed year is also classified as
    "bare" (nothing more than the fact itself — safe to drop entirely, no
    event needed) or "narrative" (comes with concrete circumstance/action or
    an interaction with another entity — the caller must keep this year
    available for Step 3, which still builds a point event from it *in
    addition to* the field already being filled here).

    Returns {"attributes": {field_name: value, ...}, "consumed_years": [...],
    "narrative_years": [...] (subset of consumed_years), "notes": str | None}."""
    candidate_fields = _attribute_candidate_fields(category)
    has_notes_field = any(f["name"] == "notes" for f in schema.get_fields(category))
    empty_result = {"attributes": {}, "consumed_years": [], "narrative_years": [], "notes": None}
    if not candidate_fields and not has_notes_field:
        return empty_result

    if candidate_fields:
        field_lines = []
        for f in candidate_fields:
            if f["type"] == "enum":
                field_lines.append(f"- {f['name']} (선택지: {', '.join(f.get('options') or [])})")
            else:
                field_lines.append(f"- {f['name']} (연도, 의미: {f.get('role')})")
        field_block = "\n".join(field_lines)
    else:
        field_block = "(해당 없음)"

    years_text = ", ".join(str(y) for y in years) if years else "(문장에 연도 없음)"
    notes_instruction = (
        '  "notes": "path 3에 해당하는 서술이 있으면 문장에서 가져온 내용, 없으면 null"\n'
        if has_notes_field
        else '  "notes": null\n'
    )
    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 신규 엔티티 등록 보조자다.\n"
        f'새로 등록되는 엔티티: "{tag}" (카테고리: {category})\n'
        f"원문: {context_sentence}\n"
        f"문장에서 추출된 연도(들): {years_text}\n\n"
        "이 문장에서 이 엔티티에 대해 서술된 내용을 아래 세 갈래 중 하나로만 분류하라 "
        "(하나의 서술은 반드시 한 갈래에만 속한다):\n\n"
        "1) lifecycle 필드값 — 아래 필드 목록에 명시적으로 대응하는 연도/선택지 값. 서술로부터 "
        "추론하거나 역산하지 마라 — 문장에 그대로 나온 값만 인정한다. 연도 값은 반드시 위 연도 "
        "목록 중 하나여야 한다. 이 값을 채웠다고 해서 2)를 무시하지는 마라 — 같은 연도에 구체적인 "
        "정황이나 상호작용이 함께 서술되어 있다면 그건 반드시 narrative_years에도 표시해야 한다 "
        "(아래 설명 참고).\n"
        f"필드 목록:\n{field_block}\n\n"
        "2) 특정 시점에 벌어진 사건이나, 시작/종료되는 상태·관계에 대한 서술 (예: '용병단이 "
        "쟝의 가입을 거절했다', '봉인되었다', '술을 먹고 싸우다가', '유리창이 터졌다') — 이건 "
        "별도의 사건 기록 파이프라인이 처리하므로 attributes에도 notes에도 넣지 마라. 대신, 이런 "
        "서술이 붙어있는 연도가 1)에서도 lifecycle 필드로 쓰였다면, 그 연도를 반드시 "
        "narrative_years에 포함시켜라 — 그래야 그 연도가 사라지지 않고 별도의 사건 기록으로도 "
        "남는다. 단순 사실 진술뿐이고(예: '100년에 태어났다'처럼 구체적 정황/상호작용이 전혀 "
        "없으면) narrative_years에 넣지 마라.\n\n"
        "3) 특정 시점과 무관하게 항상 참인 지속적 특징/규칙/성질에 대한 서술 (예: '여성만 가입 "
        "가능하다', '검은 갑옷을 입는다', '불을 두려워한다') — notes에 문장에서 가져온 내용으로 "
        "채워라. 이런 서술이 전혀 없으면 notes는 null로 둬라.\n\n"
        "아래 JSON 형식으로만 답하라 (다른 설명 금지):\n"
        "{\n"
        '  "attributes": {"필드명": 값, ...},\n'
        '  "consumed_years": [attributes에 실제로 쓰인 연도들],\n'
        '  "narrative_years": [consumed_years 중, 같은 연도에 구체적 정황/상호작용 서술이 함께 '
        '있는 것들],\n'
        f"{notes_instruction}"
        "}\n"
        '해당하는 내용이 하나도 없으면 {"attributes": {}, "consumed_years": [], '
        '"narrative_years": [], "notes": null}로 답하라.\n'
    )

    # "simple" tier (Phase 10 patch 19) — this is a mechanical 3-way sentence
    # split (lifecycle field vs. event vs. trait), not a world-consistency
    # judgment like infer_event's, so it doesn't need the reasoning tier.
    try:
        data = _extract_json(_invoke_llm(prompt, tier="simple"))
    except Exception:
        return empty_result

    valid_years = set(years)
    raw_attrs = data.get("attributes") or {}
    attributes = {}
    for name, value in raw_attrs.items():
        field_def = next((f for f in candidate_fields if f["name"] == name), None)
        if field_def is None or value is None:
            continue
        if field_def["type"] == "enum":
            if value in (field_def.get("options") or []):
                attributes[name] = value
        else:  # lifecycle year field
            try:
                year_value = int(value)
            except (TypeError, ValueError):
                continue
            if year_value in valid_years:
                attributes[name] = year_value

    consumed = {y for y in (data.get("consumed_years") or []) if y in valid_years}
    for name, value in attributes.items():
        field_def = next(f for f in candidate_fields if f["name"] == name)
        if field_def["type"] != "enum":
            consumed.add(value)

    narrative = {y for y in (data.get("narrative_years") or []) if y in consumed}

    notes_value = data.get("notes")
    notes = notes_value.strip() if has_notes_field and isinstance(notes_value, str) and notes_value.strip() else None

    return {
        "attributes": attributes,
        "consumed_years": sorted(consumed),
        "narrative_years": sorted(narrative),
        "notes": notes,
    }
