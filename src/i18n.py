"""Interface language translation (Phase 10 patch 18).

Deliberately separate from world_language (settings.py) — this is only
about the app's own menus/buttons/messages, freely toggleable at runtime
with zero risk, since it never touches stored data or what language the
AI reads/writes/reasons in.

The translation key is the literal Korean UI string itself, not an
abstract identifier — retrofitting i18n onto ~250 already-hardcoded
Korean strings across app.py, inventing a semantic key name for each
would be pure overhead with no reader benefit (the Korean text already
tells you exactly what's being displayed). `t(text, "ko")` is always a
no-op identity function; only "en" does a dictionary lookup, falling back
to the original Korean text itself if a string hasn't been added yet
(never a raw KeyError, and never a blank UI).

For strings that interpolate dynamic data (an entity id, a count, a
year), the KOREAN KEY is the template with `{}`/`{0}` placeholders — e.g.
`L("저장 완료: {0}").format(names)` — never the fully-interpolated string,
since every distinct combination of dynamic content would otherwise need
its own dictionary entry.
"""

_TRANSLATIONS: dict = {
    # --- Sidebar / top-level nav ---
    "모드": "Mode",
    "채팅": "Chat",
    "딕셔너리": "Dictionary",
    "🔍 검색 (이름 일부)": "🔍 Search (partial name)",
    "일치하는 엔티티가 없습니다.": "No matching entities.",
    "🚩 플래그 확인": "🚩 Flags",
    "플래그된 항목이 없습니다.": "No flagged items.",
    "(사유 없음)": "(no reason given)",
    "← 목록으로": "← Back to list",
    "언어 / Language": "언어 / Language",

    # --- Chat mode ---
    "채팅": "Chat",
    "일반 채팅": "Normal chat",
    "창작 모드": "Creator Mode",
    "[ ]로 엔티티를 태그하고, 연도와 함께 사건이나 설정을 입력하세요 "
    "(태그에 없는 새 이름을 쓰면 새 엔티티가 만들어집니다)":
        "Enter an event or fact with entities tagged in [ ] and a year "
        "(tag a new name to create it as a new entity)",
    "[ ]로 태그된 엔티티와 만들고 싶은 이야기를 입력하세요 "
    "(예: [쟝]과 [미라]가 원수가 되는 이야기, 2100년에)":
        "Enter the story you want, tagging entities with [ ] "
        "(e.g. [Jang] and [Mira] become enemies, in the year 2100)",
    "+ 조연 엔티티 생성 허용": "+ Allow new supporting entities",
    "체크한 카테고리에 한해 Creator가 필요하면 새로운 조연 엔티티를 만들 수 있습니다 "
    "(예: '여러 사람' 대신 '카라반 마스터 밥'). 장소/사물/세력의 기존 항목은 이 설정과 "
    "무관하게 항상 자연스럽게 활용됩니다 — 여기서는 새로 만드는 것만 켜고 끕니다.":
        "For checked categories, Creator can invent new supporting entities if the "
        "story needs one (e.g. \"Caravan Master Bob\" instead of \"some people\"). "
        "Existing locations/artifacts/factions are always usable regardless of this "
        "setting — this only toggles creating brand-new ones.",

    # --- Generic decision-dialog buttons (reused across many dialogs) ---
    "저장": "Save",
    "취소": "Cancel",
    "취소되었습니다.": "Cancelled.",
    "수정": "Edit",
    "예": "Yes",
    "아니오": "No",
    "확인": "Confirm",
    "삭제": "Delete",
    "편집": "Edit",
    "재시도": "Retry",
    "포기": "Give up",
    "돌아가기": "Go back",
    "그래도 저장": "Save anyway",
    "그대로 삭제": "Delete as-is",
    "새로 작성": "Create new",
    "저장 후 계속": "Save and continue",
    "완료되었습니다.": "Done.",
    "(비어있음)": "(empty)",
    "(없음)": "(none)",

    # --- Result / error messages ---
    "입력 오류: {0}": "Input error: {0}",
    "하드체크 결과에 따라 저장이 중단되었습니다.": "Save was blocked by a hard-check conflict.",
    "RAG 검증 결과에 따라 저장이 중단되었습니다.": "Save was blocked by a validation check.",
    "승인된 변경사항이 없어 저장할 내용이 없습니다.": "No approved changes, so there's nothing to save.",
    "엔티티가 저장되었습니다. 별도의 사건 기록은 없습니다.":
        "Entity saved. No separate event record.",
    "새로 저장할 내용이 없습니다.": "Nothing new to save.",
    "저장 완료: {0}": "Save complete: {0}",
    "{0}(갱신)": "{0} (updated)",

    # --- Entity-candidate / category-confirm dialogs ---
    "\"{0}\" 후보를 선택하세요:": "Choose a candidate for \"{0}\":",
    "\"{0}\"을(를) **{1}**(으)로 분류했습니다. 맞습니까?": "Classified \"{0}\" as **{1}**. Is that right?",
    "카테고리": "Category",
    "이름": "Name",

    # --- Terminal-status (death/end) confirmation ---
    "[{0}]가 이 사건({1}년)으로 사망(또는 활동 종료)한 것으로 추정됩니다. {2}={1}로 저장할까요?":
        "[{0}] appears to have died (or ended activity) in this event ({1}). Save {2}={1}?",
    "새로운 {0} 값": "New value for {0}",
    "{0}로 저장": "Save as {0}",

    # --- New-entity field form ---
    "[{0}] 필드를 입력하세요 (필수 항목은 *):": "Enter fields for [{0}] (required fields marked *):",

    # --- Relational-predicate confirmation ---
    "\"{0}\"라는 새로운 관계를 상태/관계 목록에 추가할까요? ({1} → {2})":
        "Add the new relation \"{0}\" to the status/relation list? ({1} → {2})",
    "근거: {0}": "Reason: {0}",
    "함께 갱신되는 엔티티: ": "Entities updated together: ",
    "새 이름으로 저장": "Save with new name",
    "이 이름으로 저장": "Save with this name",
    "새 이름": "New name",

    # --- Confirmation-needed (nothing saved) ---
    "[확인 필요] {0}": "[Confirmation needed] {0}",
    "저장된 내용이 없습니다. 입력을 나눠서 다시 시도해주세요.":
        "Nothing was saved. Please split your input and try again.",

    # --- Year-window confirm (Creator) ---
    "이야기에 사용할 연도 범위를 확인해주세요.": "Please confirm the year range to use for this story.",
    "관련 엔티티들의 존재 기간 정보가 없어 범위를 추정할 수 없습니다. 직접 입력해주세요.":
        "There's no existence-period data for the related entities, so a range can't be "
        "estimated. Please enter one directly.",
    "관련 엔티티들이 함께 존재하는 기간: {0}년 이후 (현재까지 진행 중)":
        "Period the related entities coexist: from {0} onward (still ongoing)",
    "관련 엔티티들이 함께 존재하는 기간: {0}년 ~ {1}년": "Period the related entities coexist: {0} ~ {1}",
    "시작 연도": "Start year",
    "종료 연도": "End year",

    # --- Count-mismatch (Creator) ---
    "이 이야기는 {0}개의 사건으로 구성하는 게 자연스러워 보입니다.\n\n"
    "연도 범위를 다시 입력해주시겠어요, 아니면 지정하신 {1}년 근처로 압축해서 만들까요?":
        "This story seems to naturally need {0} events.\n\nWould you like to re-enter a "
        "year range, or compress it down to around {1}?",
    "범위로 다시 입력": "Re-enter a range",
    "{0}년 근처로 압축": "Compress to around {0}",
    "이 범위로 진행": "Proceed with this range",

    # --- Relational-predicate confirmation (Creator variant, duplicated buttons already covered above) ---

    # --- Reflection-exhausted (Creator) ---
    "{0}회 시도했지만 검증을 통과하지 못했습니다.": "Tried {0} times but couldn't pass validation.",
    "마지막 반려 사유: {0}": "Reason for the last rejection: {0}",
    "마지막 시도에서 생성될 뻔한 엔티티:": "Entities that would have been created on the last attempt:",
    "마지막으로 시도된 초안:": "Last attempted draft:",
    "- [{0}, {1}년] {2}": "- [{0}, {1}] {2}",
    "그래도 검토하기": "Review anyway",

    # --- Edit-conflict (Creator) ---
    "수정한 연도가 검증에 실패했습니다: {0}": "The edited year failed validation: {0}",

    # --- Final review (Creator) ---
    "**새로 생성될 엔티티**": "**New entities to be created**",
    "**최종 검토** — 사건별로 연도를 확인/수정할 수 있습니다.":
        "**Final review** — you can check/edit the year for each event.",
    "연도": "Year",

    # --- Redo (Creator) ---
    "[Redo] — 다시 만들 때 참고할 내용이 있나요? (선택, 비워둬도 됨)":
        "[Redo] — anything to keep in mind when redrafting? (optional, can be left blank)",
    '예: "좀 더 잔인하게 해줘", "이벤트를 더 짧게 압축해줘"':
        'e.g. "make it more brutal", "compress the events shorter"',

    # --- Creator result messages ---
    "요청을 처리할 수 없습니다.": "Couldn't process the request.",
    "저장 완료: {0}개의 사건이 생성되었습니다. ({1})": "Save complete: {0} event(s) created. ({1})",
    " 새로 생성된 엔티티: {0}.": " New entities created: {0}.",

    # --- Dictionary ---
    "이 카테고리에 등록된 엔티티가 없습니다.": "No entities registered in this category.",

    # --- Entity fields / participants ---
    "**{0}** ({1}건)": "**{0}** ({1} item(s))",
    "**참가자**": "**Participants**",

    # --- Status effects panel ---
    "개인 상태 (대상 없음)": "Individual status (no target)",
    "관계형 (대상 있음)": "Relational (has a target)",
    "세계관에서 쓸 수 있는, 되돌릴 수 있는 개인 상태(수감, 저주 등)와 대상이 있는 "
    "관계형 predicate(추방, 적대 등)의 목록입니다. 새 사건 입력이나 필드 수정 화면에서 "
    "바로 선택지로 나타납니다.":
        "The list of reversible individual statuses (imprisoned, cursed, ...) and "
        "target-bearing relational predicates (exiled, hostile, ...) available in this "
        "world. These appear directly as options when entering new events or editing fields.",
    "(없음)": "(none)",
    "설명 (LLM에게 이 상태/관계가 실제로 무엇을 뜻하는지 알려줍니다)":
        "Description (tells the LLM what this status/relation actually means)",
    "설명": "Description",
    "예: 물리적으로 수감 장소를 벗어난 행동은 불가능하다.":
        "e.g. Physically leaving the place of imprisonment is impossible.",
    "설명 저장": "Save description",
    "설명을 저장했습니다.": "Description saved.",
    "현재 {0}건의 기록이 이 항목을 사용하고 있습니다. 삭제해도 그 기록 "
    "자체는 남지만, 앞으로 선택지에 나타나지 않고 관련 검증 대상에서도 "
    "빠지게 됩니다.":
        "{0} record(s) currently use this entry. Deleting it leaves those records "
        "intact, but it will no longer appear as an option or be checked against going forward.",
    "'{0}' 항목을 삭제했습니다.": "Deleted '{0}'.",
    "새 항목 추가": "Add new entry",
    "id (코드에서 predicate로 쓰일 값, 영문 권장)": "id (used as the predicate value in code, English recommended)",
    "표시 이름": "Display name",
    "유형": "Type",
    "설명 (선택, LLM에게 실제 의미를 알려줍니다)": "Description (optional, tells the LLM the real meaning)",
    "추가": "Add",
    "'{0}' ({1}) 항목을 추가했습니다.": "Added '{0}' ({1}).",

    # --- Relevant-context / field editor ---
    "관련 기록": "Related records",
    "관련성이 있어 보이는 기록이 없습니다.": "No records seem related.",
    "플래그": "Flag",
    "{0} 상세 보기": "View {0} details",
    "더 보기 ({0}건 더 있음)": "Show more ({0} more)",
    "필드 수정": "Edit field",
    "필드 선택": "Select field",
    "필드 값 검토": "Review field value",
    "하드체크 위반으로 저장할 수 없습니다:": "Can't save — hard-check violation:",
    "타임라인 충돌이 감지되지 않았습니다.": "No timeline conflicts detected.",
    "하드체크 충돌이 감지되지 않았습니다.": "No hard-check conflicts detected.",
    "저장 완료.": "Saved.",
    " {0}건 플래그 등록.": " {0} item(s) flagged.",

    # --- Delete entity ---
    "엔티티 삭제": "Delete entity",
    "이 엔티티 삭제": "Delete this entity",
    "이 엔티티가 관여한 이벤트 {0}건도 함께 정리됩니다 (다른 엔티티가 "
    "관여하지 않은 이벤트는 삭제, 관여했다면 이 엔티티의 포인터만 제거):":
        "The {0} event(s) this entity is involved in will also be cleaned up "
        "(deleted if no other entity is involved, otherwise just this entity's pointer is removed):",
    "그대로 삭제 진행": "Proceed with deletion",
    "{0} 삭제 완료.": "{0} deleted.",
    " 함께 삭제된 이벤트: {0}.": " Events deleted along with it: {0}.",
    " 포인터가 갱신된 엔티티: {0}.": " Entities with an updated pointer: {0}.",
    "취소 (유지)": "Cancel (keep)",

    # --- Timeline record edit ---
    "이벤트 수정": "Edit event",
    "장소": "Location",
    "참가자": "Participants",
    "종료 연도 (비워두면 현재도 진행 중)": "End year (leave blank if still ongoing)",
    "predicate (상태/관계 이름)": "predicate (status/relation name)",
    "주체 (entity)": "Subject (entity)",
    "대상 (target, 관계형일 때만)": "Target (relational only)",
    "비고": "Notes",
    "변경사항 검토": "Review changes",
    "이벤트 삭제": "Delete event",
    "이 이벤트 삭제": "Delete this event",
    " 포인터가 제거된 엔티티: {0}": " Entities with a pointer removed: {0}",
    "이벤트가 갱신되었습니다.": "Event updated.",

    # --- Visualization: timeline ---
    "이 엔티티와 관련된 사건이 없습니다.": "No events are related to this entity.",
    " (대상 소멸로 종료)": " (ended — subject ceased to exist)",
    " (진행중)": " (ongoing)",
    " (마지막 기록: {0}년)": " (last record: {0})",
    "{0}년, 사건: {1}": "{0}, event: {1}",
    "{0}년, {1}개 사건:": "{0}, {1} events:",
    "사건": "Event",
    "마지막 이벤트": "Last event",
    "첫 기록": "First record",

    # --- Visualization: relationship graph ---
    "{0}년에 사건이 {1}개 있습니다. 이동할 사건을 선택하세요:": "{0} has {1} events. Choose which one to go to:",
    "1-hop으로 연결된 엔티티가 없습니다.": "No 1-hop connected entities.",
    "기준 연도": "Reference year",
    "선택한 연도 기준으로 존재하는 관계가 없습니다.": "No relationships exist as of the selected year.",
    "**카테고리 필터**": "**Category filter**",
    "최소 연결 횟수": "Minimum connection count",
    "조건에 맞는 연결된 엔티티가 없습니다.": "No connected entities match the current filters.",

    # --- Entity detail top-level ---
    "알 수 없는 entity_id입니다: {0}": "Unknown entity_id: {0}",
    "존재하지 않는 엔티티입니다: {0}": "This entity doesn't exist: {0}",
    "현재 필드 값": "Current field values",
    "정보/수정": "Info / Edit",
    "시각화": "Visualization",
    "타임라인": "Timeline",
    "관계도": "Relationship graph",

    # --- Bare year suffix, used inline in several dropdown/option labels ---
    "년": "",
    "현재": "present",

    # --- main() ---
    "시각화 모드는 곧 추가될 예정입니다.": "Visualization mode is coming soon.",
}


def t(text: str, lang: str) -> str:
    """Translate `text` to `lang`. "ko" is always a no-op identity
    function (the Korean text IS the key); "en" looks it up, falling back
    to the original text itself (never a KeyError, never a blank
    display) when a string hasn't been added to the dictionary yet."""
    if lang != "en":
        return text
    return _TRANSLATIONS.get(text, text)
