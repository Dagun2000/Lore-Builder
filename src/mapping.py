"""Category inference (LLM) + entity matching/creation (rule-based) — Phase 2.

Two LLM call sites exist, both routed through `_invoke_llm` so tests can
monkeypatch a single seam instead of mocking an HTTP client:
  - infer_category: classify a tag into one of the 8 schema categories.
  - infer_terminal_status: detect whether a sentence implies a character's
    death/permanent end, to propose death_year during entity creation.
All CLI I/O goes through `_prompt` for the same reason.
"""

import re

from . import config, schema, storage

_CATEGORY_DESCRIPTIONS = {
    "character": "인물 — 이름이 있는 개별 캐릭터",
    "location": "장소 — 도시, 여관, 던전 등 물리적 공간",
    "faction": "세력/조직 — 길드, 왕국, 종교 단체 등",
    "artifact": "아이템/유물 — 무기, 보물 등 소지 가능한 물건",
    "race": "종족 — 인간, 엘프 등 생물학적 분류",
    "system": "세계관 규칙 — 마법 체계, 신성 법칙 등",
    "timeline": "사건 — 특정 시점에 벌어진 일",
    "relationship": "관계 — 두 엔티티 사이의 연결",
}
_VALID_CATEGORIES = set(_CATEGORY_DESCRIPTIONS)


# ---------------------------------------------------------------------------
# LLM / CLI seams (monkeypatched in tests)
# ---------------------------------------------------------------------------

def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.get_model("simple"), temperature=0)


def _invoke_llm(prompt: str) -> str:
    response = _get_llm().invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def _prompt(message: str) -> str:
    return input(message)


# ---------------------------------------------------------------------------
# 2-1. Category inference (LLM)
# ---------------------------------------------------------------------------

def infer_category(tag: str, context_sentence: str) -> str:
    description_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in _CATEGORY_DESCRIPTIONS.items()
    )
    prompt = (
        "너는 판타지 세계관 로어 데이터베이스의 카테고리 분류기다.\n"
        f"태그: {tag}\n"
        f"문맥 문장: {context_sentence}\n\n"
        f"카테고리 목록:\n{description_lines}\n\n"
        f"다음 중 정확히 하나만 답하라: {', '.join(_VALID_CATEGORIES)}\n"
        "카테고리 이름만 출력하고 다른 설명은 하지 마라."
    )

    raw = None
    for _ in range(2):
        raw = _invoke_llm(prompt)
        category = raw.strip().lower()
        if category in _VALID_CATEGORIES:
            return category

    raise ValueError(f"'{tag}'의 카테고리를 추론하지 못했습니다 (LLM 응답: {raw!r}).")


# ---------------------------------------------------------------------------
# 2-2. Existing entity matching (rule-based string match)
# ---------------------------------------------------------------------------

def find_existing_matches(tag: str, category: str) -> list:
    conn = storage.get_connection()
    storage.init_db(conn)
    text_fields = [f["name"] for f in schema.get_fields(category) if f["type"] == "text"]
    rows = conn.execute(f'SELECT * FROM "{category}"').fetchall()
    conn.close()

    exact_matches = []
    partial_matches = []

    for row in rows:
        entity_id = row["id"]
        id_suffix = entity_id.split("_", 1)[1] if "_" in entity_id else entity_id

        if tag == id_suffix:
            exact_matches.append(entity_id)
            continue

        haystack = " ".join(str(row[name]) for name in text_fields if row[name])
        if tag and tag in haystack:
            partial_matches.append(entity_id)

    return exact_matches if exact_matches else partial_matches


# ---------------------------------------------------------------------------
# 2-3. New-entity creation flow (character death-year proposal via LLM)
# ---------------------------------------------------------------------------

def infer_terminal_status(context_sentence: str) -> bool:
    prompt = (
        "다음 문장이 인물의 죽음이나 완전한 활동 종료(소멸, 실종 등 돌이킬 수 없는 상태)를 "
        "암시하는지 판단하라.\n"
        f"문장: {context_sentence}\n"
        "해당하면 'yes', 아니면 'no'라고만 답하라."
    )
    raw = _invoke_llm(prompt).strip().lower()
    return raw.startswith("y")


def _generate_entity_id(category: str, tag: str) -> str:
    prefix = schema.load_schema_registry()[category]["id_prefix"]
    slug = re.sub(r"\s+", "_", tag.strip())
    candidate = f"{prefix}{slug}"

    entity_id = candidate
    suffix = 1
    while storage.entity_exists(category, entity_id):
        suffix += 1
        entity_id = f"{candidate}_{suffix}"
    return entity_id


def _coerce_value(field_def: dict, raw_value: str):
    if not raw_value:
        return None
    field_type = field_def["type"]
    if field_type == "integer":
        return int(raw_value)
    if field_type == "boolean":
        return raw_value.strip().lower() in ("true", "1", "예", "y", "yes")
    if field_type == "list":
        return [v.strip() for v in raw_value.split(",") if v.strip()]
    return raw_value


def _collect_fields(
    category: str, preset: dict | None = None, allow_optional_review: bool = True
) -> dict:
    """Fill in a new entity's fields. Required fields (schema `required: true`)
    cannot be skipped or cleared — Enter is only accepted for optional fields.
    `allow_optional_review=False` skips the free-form edit loop entirely once
    required fields are satisfied (used by the character death-year fast path,
    which has no required fields of its own)."""
    fields = dict(preset or {})
    field_defs = schema.get_fields(category)
    missing_required = [
        f for f in field_defs if f.get("required") and fields.get(f["name"]) is None
    ]

    if missing_required or allow_optional_review:
        print(f"[{category}] 필드 목록 (필수 항목은 *, 비워둘 수 없음):")
        for i, f in enumerate(field_defs, start=1):
            marker = "*" if f.get("required") else " "
            current = fields.get(f["name"], "")
            print(f"  {i}.{marker} {f['name']} (현재: {current})")

    for field_def in missing_required:
        print(f"'{field_def['name']}'은(는) 필수 필드입니다. 값을 입력해야 합니다.")
        while True:
            value = _prompt(f"{field_def['name']} 값 입력 (필수): ").strip()
            if value:
                fields[field_def["name"]] = _coerce_value(field_def, value)
                break
            print("필수 필드는 비워둘 수 없습니다.")

    if not allow_optional_review:
        return fields

    while True:
        choice = _prompt("수정할 필드 번호 입력, 없으면 Enter: ").strip()
        if not choice:
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(field_defs)):
            print("잘못된 번호입니다.")
            continue
        field_def = field_defs[int(choice) - 1]
        value = _prompt(f"{field_def['name']} 값 입력: ").strip()
        if field_def.get("required") and not value:
            print(f"'{field_def['name']}'은(는) 필수 필드라 비워둘 수 없습니다.")
            continue
        fields[field_def["name"]] = _coerce_value(field_def, value)

    return fields


def _select_from_candidates(tag: str, matches: list) -> str:
    print(f"[{tag}]와(과) 일치하는 후보가 여러 개입니다:")
    for i, entity_id in enumerate(matches, start=1):
        print(f"  {i}. {entity_id}")

    while True:
        choice = _prompt(f"번호를 선택하세요 (1-{len(matches)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            return matches[int(choice) - 1]
        print("잘못된 번호입니다.")


def _create_new_entity(category: str, tag: str, context_sentence: str, year: int) -> str:
    entity_id = _generate_entity_id(category, tag)
    fields = {}
    allow_optional_review = True

    required_fields = schema.get_required_fields(category)
    if required_fields:
        names = ", ".join(f["name"] for f in required_fields)
        print(f"[{category}] 필수 필드: {names}")

    if category == "character" and infer_terminal_status(context_sentence):
        answer = _prompt(
            f"[{tag}]가 이 사건({year}년)으로 사망(또는 활동 종료)한 것으로 "
            f"추정됩니다. death_year={year}로 저장할까요? [예/아니오/수정]: "
        ).strip()
        if answer == "예":
            fields["death_year"] = year
            allow_optional_review = False
        elif answer == "아니오":
            allow_optional_review = False
        # "수정" -> falls through to full field review below.

    # Always routed through _collect_fields (not skipped) so required fields
    # on non-character categories can never be silently left empty; character
    # has no required fields, so the fast path above is unaffected.
    fields = _collect_fields(category, preset=fields, allow_optional_review=allow_optional_review)

    storage.save_entity(category, entity_id, fields)
    storage.save_to_chroma(entity_id, context_sentence, {"category": category, "tag": tag})
    print(f"[{tag}]를 신규 {category} 엔티티 {entity_id}로 저장했습니다.")
    return entity_id


def resolve_entity(tag: str, context_sentence: str, year: int) -> str:
    category = infer_category(tag, context_sentence)
    matches = find_existing_matches(tag, category)

    if len(matches) == 1:
        entity_id = matches[0]
        print(f"[{tag}]를 {entity_id}로 인식했습니다.")
        return entity_id

    if len(matches) > 1:
        return _select_from_candidates(tag, matches)

    return _create_new_entity(category, tag, context_sentence, year)
