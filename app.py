"""Lore Builder GUI — Phase 9 (Streamlit) + Phase 9 통합 패치.

Runs in-process, calling pipeline_session.py/field_update.py/flags.py
directly — no separate API server. Sidebar has a permanent search box at
the top (not a mode — it's live no matter which mode is active) and 2
mode tabs below it (chat / dictionary). "review pending" is deliberately
NOT a mode either — it's just what the entity-detail screen becomes once
you pick a field to edit, reached from either the search box or the
dictionary. Visualization (Phase 10 patch 17) isn't a top-level mode
either — it's a tab on the entity-detail screen itself (see
render_entity_detail), since it's always about one specific entity; the
sidebar used to carry a "시각화" mode placeholder for this before the
design landed here, since removed.

Widget polish (badges, styling) is explicitly out of scope for this phase —
the goal is being able to repeat every CLI test scenario through the GUI.
"""

import streamlit as st

from src import (
    creator,
    creator_session,
    deletion,
    field_update,
    flags,
    hard_check,
    i18n,
    pipeline_session,
    schema,
    storage,
    visualization,
)

_NAME_BEARING_CATEGORIES = ("character", "location", "faction", "artifact", "race")


def L(text: str) -> str:
    """Translate `text` to the current interface language (Phase 10 patch
    18) — a one-letter name because this wraps hundreds of call sites
    throughout this file; anything longer would be its own kind of
    clutter. See i18n.py for why the Korean text itself is the lookup key,
    not an abstract identifier.

    CAUTION: a handful of bare Korean strings ("예", "아니오", "그래도 저장",
    "취소", and the dict key "수정") are also literal protocol values that
    pipeline_session.py/mapping.py/main.py compare against directly when a
    decision is resumed (`if answer == "예":` etc.) — never wrap the
    ARGUMENT to `_resume(session, ...)` in `L()`; only wrap what a widget
    *displays*. Getting this backwards silently breaks that response path
    in English mode (the backend would keep expecting the Korean literal
    forever, regardless of interface language)."""
    return i18n.t(text, st.session_state.get("interface_language", "ko"))


# ---------------------------------------------------------------------------
# Session-state setup
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "interface_language": "ko",
        "chat_history": [],
        "session": None,
        "creator_history": [],
        "creator_session": None,
        "creator_new_entity_categories": {},
        "selected_entity": None,
        "_last_mode": None,
        "detail_field_name": None,
        "detail_previous_value": None,
        "detail_searched": False,
        "detail_conflicts": [],
        "detail_new_value": None,
        "detail_flag_selection": {},
        "detail_relevant_matches": [],
        "detail_relevant_show_all": False,
        "detail_confirm_delete": False,
        "dict_category_persist": None,
        "status_effect_confirm_delete": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _rollback_pending_structured_write() -> None:
    """If the field currently being reviewed is a Track A ("structured")
    field, update_field_flow's tentative write already landed in storage the
    moment "관련 기록 검색" was clicked (that write is what lets hard_check
    see the candidate value). Undo it if the user navigates away without
    clicking 저장 — otherwise switching fields/entities mid-review would
    silently leave an unconfirmed value persisted."""
    entity_id = st.session_state.get("selected_entity")
    field_name = st.session_state.get("detail_field_name")
    if not entity_id or not field_name or not st.session_state.get("detail_searched"):
        return
    category = schema.category_from_id(entity_id)
    if category and field_update.is_structured_field(category, field_name):
        storage.save_entity(category, entity_id, {field_name: st.session_state["detail_previous_value"]})


def _navigate_to_entity(entity_id) -> None:
    _rollback_pending_structured_write()
    st.session_state.detail_field_name = None
    st.session_state.detail_previous_value = None
    st.session_state.detail_searched = False
    st.session_state.detail_conflicts = []
    st.session_state.detail_flag_selection = {}
    st.session_state.detail_relevant_matches = []
    st.session_state.detail_relevant_show_all = False
    st.session_state.detail_confirm_delete = False
    st.session_state.selected_entity = entity_id


# ---------------------------------------------------------------------------
# Shared field-widget helper (Phase 9 patch B) — used by both the
# entity-detail field editor and the new-entity confirm/edit screen (patch
# A), so a type only needs to be mapped to a widget once.
# ---------------------------------------------------------------------------

def _reference_options(ref_category: str) -> list:
    """[(display_label, entity_id), ...] for a reference field's dropdown —
    the only values ever offered are real, existing entities, which is what
    keeps a free-text-typo from ever being saved into a reference field."""
    if ref_category == "status_effect":
        return [(f"{s['label']} ({s['id']})", s["id"]) for s in schema.load_status_effects()]

    if ref_category == "any":
        # relationship.subject/object can point at literally anything; no
        # single category to pull from, so offer every named entity plus
        # every timeline event by id.
        options = []
        for category in _NAME_BEARING_CATEGORIES:
            for e in storage.list_entities(category):
                options.append((f'{e.get("name") or e["id"]} ({e["id"]})', e["id"]))
        for e in storage.list_entities("timeline"):
            options.append((_year_label(e), e["id"]))
        return options

    entities = storage.list_entities(ref_category)
    if ref_category == "timeline":
        return [(_year_label(e), e["id"]) for e in entities]
    return [(f'{e.get("name") or e["id"]} ({e["id"]})', e["id"]) for e in entities]


def _year_label(event: dict) -> str:
    year = event.get("year", "?")
    return f'{event["id"]} ({year}{L("년")})'


def _render_value_field(field_def: dict, current_value, key: str, label: str | None = None):
    """type -> widget, for every schema field type. `label` overrides the
    on-screen caption (e.g. to add a required-field marker) without
    affecting which dict key the caller stores the result under — always
    use field_def["name"] for that, never the label."""
    field_type = field_def["type"]
    display_name = label if label is not None else field_def["name"]

    if field_type == "reference":
        options = _reference_options(field_def.get("ref_category"))
        labels = [opt_label for opt_label, _id in options]
        ids = [entity_id for _label, entity_id in options]
        index = ids.index(current_value) + 1 if current_value in ids else 0
        choice = st.selectbox(display_name, [L("(비어있음)")] + labels, index=index, key=key)
        return None if choice == L("(비어있음)") else ids[labels.index(choice)]

    if field_type == "enum":
        options = field_def.get("options") or []
        index = options.index(current_value) + 1 if current_value in options else 0
        choice = st.selectbox(display_name, [L("(비어있음)")] + options, index=index, key=key)
        return None if choice == L("(비어있음)") else choice

    if field_type == "boolean":
        return st.checkbox(display_name, value=bool(current_value), key=key)

    if field_type == "integer":
        # Streamlit 1.28+ lets value=None render a genuinely empty spinner
        # that stays None until the user types something — passing None
        # straight through (instead of coercing to 0) is what actually fixes
        # an unset field (birth_year, founded_year, ...) silently becoming a
        # real 0 the moment "저장" was clicked without the widget being
        # touched. (Phase 10 patch 4, G — this replaces the "값 없음"
        # checkbox from patch 3's first attempt at the same bug, now that
        # the installed Streamlit version (1.59.2) supports the native
        # option directly.)
        return st.number_input(display_name, value=current_value, step=1, key=key)

    if field_type == "list":
        raw = st.text_input(display_name, value=", ".join(current_value or []), key=key)
        return [v.strip() for v in raw.split(",") if v.strip()]

    return st.text_input(display_name, value=current_value or "", key=key)


# ---------------------------------------------------------------------------
# Chat mode — renders pipeline_session's decision types
# ---------------------------------------------------------------------------

def _resume(session, response) -> None:
    try:
        st.session_state.session = pipeline_session.resume_session(session.session_id, response)
    except ValueError:
        # A double-click on a decision button (e.g. clicking "저장" again
        # while the first click is still being processed) can deliver a
        # second response after this session already moved past the
        # decision it was answering — pipeline_session.resume_session raises
        # ValueError for that ("응답을 기다리는 결정이 없습니다"). Harmless
        # (the first click's answer already went through), just redraw the
        # current state instead of surfacing a traceback for it.
        pass
    st.rerun()


def _describe_result(result: dict) -> str:
    status = result.get("status")
    if status == "error":
        return L("입력 오류: {0}").format(result['message'])
    if status == "cancelled":
        return result.get("message", L("취소되었습니다."))
    if status == "rejected" and result.get("stage") == "hard_check":
        lines = [L("하드체크 결과에 따라 저장이 중단되었습니다.")]
        for c in result.get("conflicts", []):
            if c.severity == "blocking":
                lines.append(f"- [{c.check_type}] {c.entity_id}: {c.reason}")
        return "\n".join(lines)
    if status == "rejected" and result.get("stage") == "rag_check":
        return L("RAG 검증 결과에 따라 저장이 중단되었습니다.")
    if status == "no_changes":
        return L("승인된 변경사항이 없어 저장할 내용이 없습니다.")
    if status == "entity_only":
        return result.get("message", L("엔티티가 저장되었습니다. 별도의 사건 기록은 없습니다."))
    if status == "no_new_info":
        return result.get("message", L("새로 저장할 내용이 없습니다."))
    if status == "saved":
        applied = result.get("applied", [])
        names = ", ".join(
            L("{0}(갱신)").format(c.entity_id) if c.action == "update" else c.entity_id for c in applied
        )
        return L("저장 완료: {0}").format(names)
    return L("완료되었습니다.")


def _render_entity_candidates(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.write(L('"{0}" 후보를 선택하세요:').format(payload["tag"]))
    for i, candidate_id in enumerate(payload["candidates"]):
        if st.button(candidate_id, key=f"{key_prefix}_cand_{i}"):
            _resume(session, candidate_id)
    if payload.get("allow_create") and st.button(L("새로 작성"), key=f"{key_prefix}_create"):
        _resume(session, pipeline_session.CREATE_NEW)


def _render_entity_category_and_name(session, decision, key_prefix) -> None:
    """Phase 9 patch A: category confirmation is the headline here, not the
    name — a wrong category (person mistaken for an item) is the expensive
    mistake; the LLM rarely gets the name wrong."""
    payload = decision.payload
    categories = payload["categories"]
    st.write(
        L('"{0}"을(를) **{1}**(으)로 분류했습니다. 맞습니까?')
        .format(payload["tag"], payload["inferred_category"])
    )
    category = st.selectbox(
        L("카테고리"),
        categories,
        index=categories.index(payload["inferred_category"]),
        key=f"{key_prefix}_category",
    )

    name = None
    # Re-derive has_name_field for whatever category is *currently selected*
    # in the box, not just the originally-inferred one — switching category
    # changes whether a name field even exists.
    has_name_field = "name" in {f["name"] for f in schema.get_fields(category)}
    if has_name_field:
        name = st.text_input(L("이름"), value=payload["default_name"], key=f"{key_prefix}_name")

    col1, col2, col3 = st.columns(3)
    if col1.button(L("저장 후 계속"), key=f"{key_prefix}_save"):
        _resume(session, {"category": category, "name": name, "action": "save"})
    if col2.button(L("편집"), key=f"{key_prefix}_edit"):
        _resume(session, {"category": category, "name": name, "action": "edit"})
    if col3.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume(session, {"category": category, "name": name, "action": "cancel"})


def _render_entity_terminal_status(session, decision, key_prefix) -> None:
    payload = decision.payload
    field_name = payload.get("field_name", "death_year")
    st.write(
        L("[{0}]가 이 사건({1}년)으로 사망(또는 활동 종료)한 것으로 추정됩니다. {2}={1}로 저장할까요?")
        .format(payload["tag"], payload["year"], field_name)
    )
    col1, col2, col3 = st.columns(3)
    # Button LABELS are translated; the values passed to _resume() below are
    # NOT — pipeline_session.py/mapping.py/main.py compare against these
    # exact Korean strings as a fixed internal protocol, regardless of
    # what the button displays (see the module-level note near L()).
    if col1.button(L("예"), key=f"{key_prefix}_yes"):
        _resume(session, "예")
    if col2.button(L("아니오"), key=f"{key_prefix}_no"):
        _resume(session, "아니오")
    if col3.button(L("수정"), key=f"{key_prefix}_edit_toggle"):
        st.session_state[f"{key_prefix}_editing"] = True
    if st.session_state.get(f"{key_prefix}_editing"):
        new_year = st.number_input(
            L("새로운 {0} 값").format(field_name), value=payload["year"], step=1, key=f"{key_prefix}_year"
        )
        if st.button(L("{0}로 저장").format(field_name), key=f"{key_prefix}_edit_confirm"):
            _resume(session, {"수정": {field_name: int(new_year)}})


def _render_entity_required_field(session, decision, key_prefix) -> None:
    """Full field form (required forced server-side, optional included) —
    every field goes through the same type -> widget mapping as the
    entity-detail editor (Phase 9 patch B)."""
    payload = decision.payload
    st.write(L("[{0}] 필드를 입력하세요 (필수 항목은 *):").format(payload["category"]))
    values = {}
    for f in payload["fields"]:
        label = f"{f['name']} *" if f["required"] else f["name"]
        values[f["name"]] = _render_value_field(
            f, None, key=f"{key_prefix}_{f['name']}", label=label
        )
    if st.button(L("저장"), key=f"{key_prefix}_submit"):
        _resume(session, values)


def _render_hard_check_warning(session, decision, key_prefix) -> None:
    """Phase 10 patch 7 (E): "수정" used to sit between these two buttons
    but never actually offered any editing — it just fell through to the
    same rejection "그래도 저장"'s absence already causes. Two honest
    options instead of three, one of which lied about what it did."""
    payload = decision.payload
    st.warning(f"[{payload['entity_id']}] {payload['reason']}")
    col1, col2 = st.columns(2)
    # See the L() docstring: label translated, protocol value untouched.
    if col1.button(L("그래도 저장"), key=f"{key_prefix}_accept"):
        _resume(session, "그래도 저장")
    if col2.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume(session, "취소")


def _render_rag_judgment(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.warning(f"[{payload['judgment_type']}] {payload['reason']}")
    col1, col2 = st.columns(2)
    # See the L() docstring: label translated, protocol value untouched.
    if col1.button(L("그래도 저장"), key=f"{key_prefix}_accept"):
        _resume(session, "그래도 저장")
    if col2.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume(session, "취소")


def _render_new_relational_predicate(session, decision, key_prefix) -> None:
    """Phase 10 patch 16, A: Step 3 proposed a target-bearing (relational)
    predicate not yet in status_effects.yaml — 저장 registers it as-is,
    수정 (two-step, same pattern as _render_entity_terminal_status) lets the
    user rename it first, 취소 drops just this duration record (any other
    record from the same input, e.g. a point event, still saves
    independently)."""
    payload = decision.payload
    st.write(
        L('"{0}"라는 새로운 관계를 상태/관계 목록에 추가할까요? ({1} → {2})')
        .format(payload["predicate"], payload.get("entity_id"), payload.get("target_id"))
    )
    if payload.get("reason"):
        st.caption(payload["reason"])

    col1, col2, col3 = st.columns(3)
    if col1.button(L("저장"), key=f"{key_prefix}_save"):
        _resume(session, {"action": "save"})
    if col2.button(L("수정"), key=f"{key_prefix}_edit_toggle"):
        st.session_state[f"{key_prefix}_editing"] = True
        st.rerun()
    if col3.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume(session, {"action": "cancel"})

    if st.session_state.get(f"{key_prefix}_editing"):
        new_name = st.text_input(L("새 이름"), value=payload["predicate"], key=f"{key_prefix}_name")
        if st.button(L("이 이름으로 저장"), key=f"{key_prefix}_edit_confirm"):
            _resume(session, {"action": "edit", "name": new_name})


def _render_diff_review(session, decision, key_prefix) -> None:
    """One bundled decision for the whole diff (Phase 10 patch) — the
    primary record plus whichever other entities get an event_ids/cache
    update alongside it, shown as information only. No per-item toggling,
    no edit here: 저장 applies everything, 취소 applies nothing."""
    payload = decision.payload
    st.write(f"**{payload['action'].upper()} {payload['category']}**: {payload['entity_id']}")
    st.caption(L("근거: {0}").format(payload['reason']))
    st.json(payload["fields"])
    if payload["affected_entities"]:
        st.caption(L("함께 갱신되는 엔티티: ") + ", ".join(payload["affected_entities"]))
    col1, col2 = st.columns(2)
    if col1.button(L("저장"), key=f"{key_prefix}_save"):
        _resume(session, True)
    if col2.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume(session, False)


def _render_multi_event_warning(session, decision, key_prefix) -> None:
    """Nothing gets saved here either way (see pipeline_session's identical
    comment) — this is purely an acknowledgment, not a choice between two
    outcomes that both do the same non-thing."""
    payload = decision.payload
    st.warning(L("[확인 필요] {0}").format(payload['reason']))
    st.caption(L("저장된 내용이 없습니다. 입력을 나눠서 다시 시도해주세요."))
    if st.button(L("확인"), key=f"{key_prefix}_ack"):
        _resume(session, None)


_DECISION_RENDERERS = {
    "entity_candidates": _render_entity_candidates,
    "entity_category_and_name": _render_entity_category_and_name,
    "entity_terminal_status": _render_entity_terminal_status,
    "entity_required_field": _render_entity_required_field,
    "multi_event_warning": _render_multi_event_warning,
    "new_relational_predicate": _render_new_relational_predicate,
    "hard_check_warning": _render_hard_check_warning,
    "rag_judgment": _render_rag_judgment,
    "diff_review": _render_diff_review,
}


def render_chat_mode() -> None:
    st.header(L("채팅"))

    # Explicit mode toggle, not pattern detection (Phase 10 patch 22, A) —
    # a selectbox directly above the chat input is the closest native
    # Streamlit proxy to "inside the input box" (chat_input can't embed
    # another widget literally inside it). The two modes branch into
    # completely separate entry functions below; there is no shared
    # guessing logic anywhere for which one an input "looks like".
    # Translated options built up front + reverse-mapped back to the raw
    # value, not format_func=L — see main()'s mode radio for why: a
    # format_func that reads st.session_state breaks Streamlit's own
    # AppTest harness (it invokes format_func outside any live script
    # context on the run after a language switch).
    _chat_mode_options = ["일반 채팅", "창작 모드"]
    _chat_mode_display = [L(m) for m in _chat_mode_options]
    chat_mode_choice = st.selectbox(
        L("모드"), _chat_mode_display, key="chat_mode_toggle", label_visibility="collapsed"
    )
    is_creator = _chat_mode_options[_chat_mode_display.index(chat_mode_choice)] == "창작 모드"
    if is_creator:
        render_creator_mode()
        return

    text = st.chat_input(
        L("[ ]로 엔티티를 태그하고, 연도와 함께 사건이나 설정을 입력하세요 "
          "(태그에 없는 새 이름을 쓰면 새 엔티티가 만들어집니다)")
    )
    if text:
        st.session_state.chat_history.append({"role": "user", "content": text})
        st.session_state.session = pipeline_session.start_session(text)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    session = st.session_state.session
    if session is None:
        return

    if session.pending_decision is not None:
        decision = session.pending_decision
        key_prefix = f"{session.session_id}_{id(decision)}"
        with st.chat_message("assistant"):
            _DECISION_RENDERERS[decision.decision_type](session, decision, key_prefix)
    else:
        description = _describe_result(session.result)
        with st.chat_message("assistant"):
            st.write(description)
        st.session_state.chat_history.append({"role": "assistant", "content": description})
        st.session_state.session = None


# ---------------------------------------------------------------------------
# Creator mode — renders creator_session's decision types (Phase 10 patch 22)
# ---------------------------------------------------------------------------

def _resume_creator(session, response) -> None:
    try:
        st.session_state.creator_session = creator_session.resume_session(session.session_id, response)
    except ValueError:
        # Same double-click guard as _resume above.
        pass
    st.rerun()


def _render_creator_entity_candidates(session, decision, key_prefix) -> None:
    """Same decision_type/payload shape as pipeline_session's own tag
    disambiguation, reused for GUI consistency — Creator always sends
    allow_create=False, so there's no "신규 작성" branch to render here."""
    payload = decision.payload
    st.write(L('"{0}" 후보를 선택하세요:').format(payload["tag"]))
    for i, candidate_id in enumerate(payload["candidates"]):
        if st.button(candidate_id, key=f"{key_prefix}_cand_{i}"):
            _resume_creator(session, candidate_id)


def _render_creator_year_confirm(session, decision, key_prefix) -> None:
    payload = decision.payload
    lower, upper = payload["lower"], payload["upper"]
    st.write(L("이야기에 사용할 연도 범위를 확인해주세요."))
    if lower is None and upper is None:
        st.caption(L("관련 엔티티들의 존재 기간 정보가 없어 범위를 추정할 수 없습니다. 직접 입력해주세요."))
    elif upper is None:
        st.caption(L("관련 엔티티들이 함께 존재하는 기간: {0}년 이후 (현재까지 진행 중)").format(lower))
    else:
        st.caption(L("관련 엔티티들이 함께 존재하는 기간: {0}년 ~ {1}년").format(lower, upper))

    default_lower = lower if lower is not None else 0
    default_upper = upper if upper is not None else default_lower + 50
    new_lower = st.number_input(L("시작 연도"), value=default_lower, step=1, key=f"{key_prefix}_lower")
    new_upper = st.number_input(L("종료 연도"), value=default_upper, step=1, key=f"{key_prefix}_upper")

    col1, col2 = st.columns(2)
    if col1.button(L("확인"), key=f"{key_prefix}_confirm"):
        _resume_creator(session, {"action": "confirm", "lower": int(new_lower), "upper": int(new_upper)})
    if col2.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume_creator(session, {"action": "cancel"})


def _render_creator_count_mismatch(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.write(
        L("이 이야기는 {0}개의 사건으로 구성하는 게 자연스러워 보입니다.\n\n"
          "연도 범위를 다시 입력해주시겠어요, 아니면 지정하신 {1}년 근처로 압축해서 만들까요?")
        .format(payload['natural_event_count'], payload['year'])
    )
    col1, col2, col3 = st.columns(3)
    if col1.button(L("범위로 다시 입력"), key=f"{key_prefix}_widen_toggle"):
        st.session_state[f"{key_prefix}_widening"] = True
        st.rerun()
    if col2.button(L("{0}년 근처로 압축").format(payload['year']), key=f"{key_prefix}_compress"):
        _resume_creator(session, {"action": "compress"})
    if col3.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume_creator(session, {"action": "cancel"})

    if st.session_state.get(f"{key_prefix}_widening"):
        new_lower = st.number_input(L("시작 연도"), value=payload["year"], step=1, key=f"{key_prefix}_lower")
        new_upper = st.number_input(L("종료 연도"), value=payload["year"] + 10, step=1, key=f"{key_prefix}_upper")
        if st.button(L("이 범위로 진행"), key=f"{key_prefix}_widen_confirm"):
            _resume_creator(session, {"action": "widen", "lower": int(new_lower), "upper": int(new_upper)})


def _render_creator_new_relational_predicate(session, decision, key_prefix) -> None:
    """Same decision_type/payload shape as pipeline_session's patch-16
    registry gate, reused for GUI consistency."""
    payload = decision.payload
    st.write(
        L('"{0}"라는 새로운 관계를 상태/관계 목록에 추가할까요? ({1} → {2})')
        .format(payload["predicate"], payload.get("entity_id"), payload.get("target_id"))
    )
    if payload.get("reason"):
        st.caption(payload["reason"])

    col1, col2, col3 = st.columns(3)
    if col1.button(L("저장"), key=f"{key_prefix}_save"):
        _resume_creator(session, {"action": "save"})
    if col2.button(L("수정"), key=f"{key_prefix}_edit_toggle"):
        st.session_state[f"{key_prefix}_editing"] = True
        st.rerun()
    if col3.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume_creator(session, {"action": "cancel"})

    if st.session_state.get(f"{key_prefix}_editing"):
        new_name = st.text_input(L("새 이름"), value=payload["predicate"], key=f"{key_prefix}_name")
        if st.button(L("이 이름으로 저장"), key=f"{key_prefix}_edit_confirm"):
            _resume_creator(session, {"action": "edit", "name": new_name})


def _render_creator_exhausted(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.warning(L("{0}회 시도했지만 검증을 통과하지 못했습니다.").format(payload['attempts']))
    st.caption(L("마지막 반려 사유: {0}").format(payload['reason']))
    new_entities = payload.get("new_entities") or []
    if new_entities:
        st.write(L("마지막 시도에서 생성될 뻔한 엔티티:"))
        for ent in new_entities:
            st.write(f"- {ent['fields'].get('name') or ent['entity_id']} ({ent['category']})")
    st.write(L("마지막으로 시도된 초안:"))
    for e in payload["events"]:
        year = e["year"] if e["event_type"] == "point" else e["start_year"]
        st.write(L("- [{0}, {1}년] {2}").format(e['event_type'], year, e['notes']))
    col1, col2 = st.columns(2)
    if col1.button(L("그래도 검토하기"), key=f"{key_prefix}_keep"):
        _resume_creator(session, {"action": "keep_anyway"})
    if col2.button(L("포기"), key=f"{key_prefix}_discard"):
        _resume_creator(session, {"action": "discard"})


def _render_creator_edit_conflict(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.warning(L("수정한 연도가 검증에 실패했습니다: {0}").format(payload['reason']))
    col1, col2 = st.columns(2)
    if col1.button(L("그래도 저장"), key=f"{key_prefix}_accept"):
        _resume_creator(session, {"action": "save_anyway"})
    if col2.button(L("돌아가기"), key=f"{key_prefix}_back"):
        _resume_creator(session, {"action": "back"})


def _render_creator_final_review(session, decision, key_prefix) -> None:
    """Which year field(s) are editable depends on the duration event's own
    action (Phase 10 patch 22 follow-up) — a "clear" (e.g. lifting an
    exile/release from prison) only ever carries end_year; start_year
    belongs to the *original* record being closed, not this one, and is
    legitimately None here. Rendering a start_year box unconditionally for
    every duration event crashed on int(None) the moment a clear action
    showed up in a real draft."""
    payload = decision.payload
    new_entities = payload.get("new_entities") or []
    if new_entities:
        st.write(L("**새로 생성될 엔티티**"))
        for ent in new_entities:
            label = ent["fields"].get("name") or ent["entity_id"]
            st.write(f"- {label} ({ent['category']})")
            if ent["fields"].get("notes"):
                st.caption(ent["fields"]["notes"])
        st.divider()

    st.write(L("**최종 검토** — 사건별로 연도를 확인/수정할 수 있습니다."))
    year_widgets = {}
    for e in payload["events"]:
        idx = e["index"]
        st.write(f"{idx + 1}. [{e['event_type']}] {e['notes']}")
        entries = []
        if e["event_type"] == "point":
            new_year = st.number_input(L("연도"), value=e["year"], step=1, key=f"{key_prefix}_year_{idx}")
            entries.append(("year", e["year"], new_year))
        else:
            action = (e.get("duration_effect") or {}).get("action", "set")
            if action in ("set", "set_closed"):
                new_start = st.number_input(
                    L("시작 연도"), value=e["start_year"], step=1, key=f"{key_prefix}_start_{idx}"
                )
                entries.append(("start_year", e["start_year"], new_start))
            if action in ("clear", "set_closed"):
                new_end = st.number_input(
                    L("종료 연도"), value=e["end_year"], step=1, key=f"{key_prefix}_end_{idx}"
                )
                entries.append(("end_year", e["end_year"], new_end))
        year_widgets[idx] = entries
        st.divider()

    col1, col2, col3 = st.columns(3)
    if col1.button(L("저장"), key=f"{key_prefix}_save"):
        edits = {}
        for idx, entries in year_widgets.items():
            idx_edits = {}
            for field_name, original, new_value in entries:
                # A blank spinner (never touched, or cleared by the user)
                # comes back as None -- treat as "no change" rather than
                # trying to save a null year onto a field that requires one.
                if new_value is not None and new_value != original:
                    idx_edits[field_name] = int(new_value)
            if idx_edits:
                edits[str(idx)] = idx_edits
        _resume_creator(session, {"action": "save", "year_edits": edits})
    if col2.button(L("취소"), key=f"{key_prefix}_cancel"):
        _resume_creator(session, {"action": "cancel"})
    if col3.button("Redo", key=f"{key_prefix}_redo_toggle"):
        st.session_state[f"{key_prefix}_redoing"] = True
        st.rerun()

    if st.session_state.get(f"{key_prefix}_redoing"):
        supplement = st.text_input(
            L("[Redo] — 다시 만들 때 참고할 내용이 있나요? (선택, 비워둬도 됨)"),
            placeholder=L('예: "좀 더 잔인하게 해줘", "이벤트를 더 짧게 압축해줘"'),
            key=f"{key_prefix}_supplement",
        )
        if st.button(L("재시도"), key=f"{key_prefix}_redo_confirm"):
            _resume_creator(session, {"action": "redo", "supplement": supplement})


_CREATOR_DECISION_RENDERERS = {
    "entity_candidates": _render_creator_entity_candidates,
    "creator_year_confirm": _render_creator_year_confirm,
    "creator_count_mismatch": _render_creator_count_mismatch,
    "new_relational_predicate": _render_creator_new_relational_predicate,
    "creator_exhausted": _render_creator_exhausted,
    "creator_final_review": _render_creator_final_review,
    "creator_edit_conflict": _render_creator_edit_conflict,
}


def _describe_creator_result(result: dict) -> str:
    status = result.get("status")
    if status == "error":
        return L("입력 오류: {0}").format(result['message'])
    if status == "rejected":
        return result.get("message", L("요청을 처리할 수 없습니다."))
    if status == "cancelled":
        return result.get("message") or L("취소되었습니다.")
    if status == "saved":
        applied = result.get("applied", [])
        creates = [c.entity_id for c in applied if c.action == "create" and c.category == "timeline"]
        new_entities = [c.entity_id for c in applied if c.action == "create" and c.category != "timeline"]
        message = L("저장 완료: {0}개의 사건이 생성되었습니다. ({1})").format(len(creates), ', '.join(creates))
        if new_entities:
            message += L(" 새로 생성된 엔티티: {0}.").format(', '.join(new_entities))
        return message
    return L("완료되었습니다.")


def _render_new_entity_checkboxes() -> set:
    """One checkbox per eligible category (Phase 10 patch 22, B), generated
    from schema.list_categories() — not a hardcoded list, so a category
    added later shows up automatically. Off by default, matching the
    checkboxes' own off-by-default intent: supporting-entity creation is
    something opted into per-request, not a standing behavior.

    Backed by st.session_state.creator_new_entity_categories (a plain dict,
    not the checkboxes' own widget keys) — confirmed via direct testing
    that a checkbox's own key-based session_state value gets cleared the
    moment it isn't rendered for one script run (e.g. switching to
    딕셔너리 mode and back), which reset every checkbox to unchecked on
    return. Explicitly passing value=stored.get(category, False) on every
    render, and writing the result back to the same persisted dict, keeps
    the checked state alive across mode switches regardless of what
    Streamlit does with the widget's own key in between."""
    stored = st.session_state.creator_new_entity_categories
    allowed = set()
    with st.expander(L("+ 조연 엔티티 생성 허용")):
        st.caption(
            L("체크한 카테고리에 한해 Creator가 필요하면 새로운 조연 엔티티를 만들 수 있습니다 "
              "(예: '여러 사람' 대신 '카라반 마스터 밥'). 장소/사물/세력의 기존 항목은 이 설정과 "
              "무관하게 항상 자연스럽게 활용됩니다 — 여기서는 새로 만드는 것만 켜고 끕니다.")
        )
        for category in creator.eligible_categories():
            checked = st.checkbox(
                category, value=stored.get(category, False), key=f"creator_new_{category}"
            )
            stored[category] = checked
            if checked:
                allowed.add(category)
    return allowed


def render_creator_mode() -> None:
    allowed_new_categories = _render_new_entity_checkboxes()
    text = st.chat_input(
        L("[ ]로 태그된 엔티티와 만들고 싶은 이야기를 입력하세요 "
          "(예: [쟝]과 [미라]가 원수가 되는 이야기, 2100년에)")
    )
    if text:
        st.session_state.creator_history.append({"role": "user", "content": text})
        st.session_state.creator_session = creator_session.start_session(text, allowed_new_categories)

    for msg in st.session_state.creator_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    session = st.session_state.creator_session
    if session is None:
        return

    if session.pending_decision is not None:
        decision = session.pending_decision
        key_prefix = f"{session.session_id}_{id(decision)}"
        with st.chat_message("assistant"):
            _CREATOR_DECISION_RENDERERS[decision.decision_type](session, decision, key_prefix)
    else:
        description = _describe_creator_result(session.result)
        with st.chat_message("assistant"):
            st.write(description)
        st.session_state.creator_history.append({"role": "assistant", "content": description})
        st.session_state.creator_session = None


# ---------------------------------------------------------------------------
# Sidebar search — always visible, not a mode
# ---------------------------------------------------------------------------

def render_sidebar_search() -> None:
    """Lives at the top of the sidebar regardless of which mode tab is
    active — searching and switching modes are independent actions, not
    alternatives, so this was pulled out of the mode dispatch entirely."""
    query = st.sidebar.text_input(L("🔍 검색 (이름 일부)"), key="global_search")
    if not query:
        return

    results = []
    for category in _NAME_BEARING_CATEGORIES:
        for entity in storage.list_entities(category):
            if query in (entity.get("name") or ""):
                results.append((category, entity))

    if not results:
        st.sidebar.caption(L("일치하는 엔티티가 없습니다."))
        return

    for category, entity in results:
        label = f"[{category}] {entity['name']} ({entity['id']})"
        if st.sidebar.button(label, key=f"search_{entity['id']}"):
            _navigate_to_entity(entity["id"])
            st.rerun()


def _dictionary_label(category: str, entity: dict) -> str:
    """Phase 10 patch 8: the id is a content-derived slug (often just the
    name/notes with spaces swapped for underscores) — never useful for a
    person to look at, so it's dropped everywhere except as a fallback for
    the one category with no `name` field at all (timeline)."""
    if category == "timeline":
        if entity.get("year") is not None:
            return f"[{entity['year']}{L('년')}] {entity.get('notes') or entity['id']}"
        start = entity.get("start_year")
        span = f"{start}~{entity.get('end_year') or L('현재')}" if start is not None else "?"
        predicate = entity.get("predicate") or ""
        return f"[{span}] {predicate} — {entity.get('notes') or entity['id']}"
    return entity.get("name") or entity["id"]


def _entity_display_label(entity_id: str) -> str:
    """_dictionary_label for a single id looked up from scratch — used
    wherever a reference/pointer value needs a human-readable caption
    (Phase 10 patch 20). Falls back to the bare id for a dangling reference
    (the pointed-at row no longer exists) rather than erroring."""
    category = schema.category_from_id(entity_id)
    entity = storage.get_entity(category, entity_id) if category else None
    return _dictionary_label(category, entity) if entity else entity_id


def _render_entity_link(entity_id: str, key: str) -> None:
    """A navigation button labeled with the target's own display name
    (Phase 10 patch 20) — the shared building block for turning a reference
    field or an event_ids entry into something clickable instead of a raw
    id sitting in a JSON dump."""
    if st.button(_entity_display_label(entity_id), key=key):
        _navigate_to_entity(entity_id)
        st.rerun()


def _entity_exists(entity_id: str) -> bool:
    category = schema.category_from_id(entity_id)
    return bool(category and storage.get_entity(category, entity_id))


def _render_entity_fields(category: str, entity_id: str, entity: dict) -> None:
    """Replaces the old raw st.json(entity) dump (Phase 10 patch 20):
    reference fields (race, location, ...) and event_ids become clickable
    navigation buttons via _render_entity_link, everything else is plain
    "필드명: 값" text. Empty/null fields are skipped, same as st.json would
    implicitly show them (as null) but with no more useful information.

    event_ids is filtered to entries that still resolve to a real stored
    record before rendering. A dangling pointer (the record it points at
    was deleted without this entity's own event_ids being updated to
    match — e.g. a hand-edited seed file, since nothing auto-syncs those,
    see the README's "editing the seed world" section) would otherwise
    still render as a clickable button showing nothing but its own raw
    id, since _entity_display_label falls back to that rather than
    erroring. That fallback is the right behavior for a single reference
    field, but a whole redundant, meaningless entry in a list is better
    just not shown at all."""
    for field_def in schema.get_fields(category):
        name = field_def["name"]
        value = entity.get(name)
        if value in (None, "", []):
            continue

        if field_def["type"] == "reference":
            st.write(f"**{name}**")
            _render_entity_link(value, key=f"reflink_{entity_id}_{name}")
        elif name == "event_ids":
            live_ids = [eid for eid in value if _entity_exists(eid)]
            if not live_ids:
                continue
            st.write(L("**{0}** ({1}건)").format(name, len(live_ids)))
            for i, eid in enumerate(live_ids):
                _render_entity_link(eid, key=f"reflink_{entity_id}_{name}_{i}")
        else:
            st.write(f"**{name}**: {value}")


def _render_timeline_participants(entity_id: str, entity: dict) -> None:
    """A point event doesn't store its own participant list as a field —
    who's involved is only knowable via the reverse lookup (same one
    _render_timeline_detail already uses to seed its "참가자" multiselect),
    so _render_entity_fields' generic schema-field loop can't surface it.
    A duration event needs nothing extra here: its entity/target are real
    reference fields and already get links from _render_entity_fields."""
    is_point = entity.get("year") is not None or entity.get("entity") is None
    if not is_point:
        return
    # The reverse lookup also catches the location itself (it points back
    # at this event_id the same way a character or artifact does) — already
    # shown separately as "장소" above, so it's filtered out here rather
    # than changed at the storage layer, which other code may depend on.
    location_id = entity.get("location")
    participants = [
        eid for _cat, eid in storage.find_entities_referencing_event(entity_id) if eid != location_id
    ]
    if not participants:
        return
    st.write(L("**참가자**"))
    for i, eid in enumerate(participants):
        _render_entity_link(eid, key=f"reflink_{entity_id}_participants_{i}")


_STATUS_EFFECTS_PSEUDO_CATEGORY = "Relations/Status"

_STATUS_EFFECT_TYPE_LABELS = {
    "individual": "개인 상태 (대상 없음)",
    "relational": "관계형 (대상 있음)",
}


def _status_effect_usage_count(effect_id: str, effect_type: str) -> int:
    """How many timeline duration records currently use effect_id as their
    predicate — shown before deletion so removing one doesn't silently
    orphan existing records from every check/dropdown that reads
    status_effects.yaml going forward. Phase 10 patch 16: individual
    (target-less) and relational (target-bearing) entries are counted by
    the matching side of that same target-presence signal."""
    matches_target = (lambda t: t is None) if effect_type == "individual" else (lambda t: t is not None)
    return sum(
        1
        for e in storage.list_entities("timeline")
        if e.get("predicate") == effect_id and matches_target(e.get("target"))
    )


def _render_status_effects_panel() -> None:
    """A GUI-editable status_effects.yaml (Phase 10 patch 14, extended
    patch 16 with a type split) — the set of reversible statuses
    (imprisoned, cursed, ...) *and* target-bearing relational predicates
    (exiled, enemy_of, ...) a world can have is a setting-specific choice,
    not something the code should hardcode; a sci-fi setting might add
    "cryosleep" the same way a fantasy one added "imprisoned", and a new
    relational fact (patch 16, A) grows this list automatically the first
    time Step 3 proposes one that isn't here yet. Lives as a pseudo-category
    in the dictionary rather than a real schema_registry.yaml category,
    since these aren't entities — they never get their own id/detail
    screen, just this list."""
    st.write(
        L("세계관에서 쓸 수 있는, 되돌릴 수 있는 개인 상태(수감, 저주 등)와 대상이 있는 "
          "관계형 predicate(추방, 적대 등)의 목록입니다. 새 사건 입력이나 필드 수정 화면에서 "
          "바로 선택지로 나타납니다.")
    )

    effects = schema.load_status_effects()
    pending = st.session_state.status_effect_confirm_delete

    for effect_type, type_label in _STATUS_EFFECT_TYPE_LABELS.items():
        st.subheader(L(type_label))
        typed_effects = [e for e in effects if e.get("type", "individual") == effect_type]
        if not typed_effects:
            st.caption(L("(없음)"))

        for effect in typed_effects:
            effect_id = effect["id"]
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{effect_id}** ({effect['label']})")

            with st.expander(L("설명 (LLM에게 이 상태/관계가 실제로 무엇을 뜻하는지 알려줍니다)")):
                notes_key = f"status_effect_notes_{effect_id}"
                new_notes = st.text_area(
                    L("설명"), value=effect.get("notes") or "", key=notes_key,
                    placeholder=L("예: 물리적으로 수감 장소를 벗어난 행동은 불가능하다."),
                )
                if st.button(L("설명 저장"), key=f"status_effect_notes_save_{effect_id}"):
                    schema.update_status_effect_notes(effect_id, new_notes)
                    st.success(L("설명을 저장했습니다."))
                    st.rerun()

            if pending != effect_id:
                if col2.button(L("삭제"), key=f"status_effect_delete_{effect_id}"):
                    st.session_state.status_effect_confirm_delete = effect_id
                    st.rerun()
                continue

            usage = _status_effect_usage_count(effect_id, effect_type)
            if usage:
                st.warning(
                    L("현재 {0}건의 기록이 이 항목을 사용하고 있습니다. 삭제해도 그 기록 "
                      "자체는 남지만, 앞으로 선택지에 나타나지 않고 관련 검증 대상에서도 "
                      "빠지게 됩니다.").format(usage)
                )
            confirm_col1, confirm_col2 = st.columns(2)
            if confirm_col1.button(L("그대로 삭제"), key=f"status_effect_delete_confirm_{effect_id}"):
                schema.remove_status_effect(effect_id)
                st.session_state.status_effect_confirm_delete = None
                st.success(L("'{0}' 항목을 삭제했습니다.").format(effect_id))
                st.rerun()
            if confirm_col2.button(L("취소"), key=f"status_effect_delete_cancel_{effect_id}"):
                st.session_state.status_effect_confirm_delete = None
                st.rerun()

    st.divider()
    st.subheader(L("새 항목 추가"))
    with st.form("status_effect_add_form", clear_on_submit=True):
        new_id = st.text_input(L("id (코드에서 predicate로 쓰일 값, 영문 권장)"), key="status_effect_new_id")
        new_label = st.text_input(L("표시 이름"), key="status_effect_new_label")
        new_type = st.selectbox(
            L("유형"), [L(v) for v in _STATUS_EFFECT_TYPE_LABELS.values()], key="status_effect_new_type"
        )
        new_notes = st.text_area(
            L("설명 (선택, LLM에게 실제 의미를 알려줍니다)"), key="status_effect_new_notes",
            placeholder=L("예: 물리적으로 수감 장소를 벗어난 행동은 불가능하다."),
        )
        submitted = st.form_submit_button(L("추가"), key="status_effect_add")
    if submitted:
        # new_type is whatever the selectbox displayed (translated), so the
        # reverse lookup must translate each candidate the same way before
        # comparing — matching against the raw Korean label directly would
        # silently never match in English mode.
        type_value = next(k for k, v in _STATUS_EFFECT_TYPE_LABELS.items() if L(v) == new_type)
        try:
            schema.add_status_effect(new_id, new_label, type_value, notes=new_notes)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(L("'{0}' ({1}) 항목을 추가했습니다.").format(new_id, new_label))
            st.rerun()


def render_dictionary_mode() -> None:
    """Phase 10 patch 7 (G): Streamlit drops a widget's Session State entry
    at the end of any run where that widget wasn't instantiated — and this
    selectbox isn't instantiated at all while an entity detail screen is
    open (render_entity_detail returns before render_dictionary_mode ever
    runs), so "dict_category" was silently wiped every time, and coming
    back here always re-created the selectbox at its default (index 0,
    "character"). dict_category_persist lives outside any widget's
    lifecycle, so it survives the round trip and can drive `index=` here."""
    st.header(L("딕셔너리"))
    categories = schema.list_categories() + [_STATUS_EFFECTS_PSEUDO_CATEGORY]
    remembered = st.session_state.dict_category_persist
    default_index = categories.index(remembered) if remembered in categories else 0
    category = st.selectbox(L("카테고리"), categories, index=default_index, key="dict_category")
    st.session_state.dict_category_persist = category

    if category == _STATUS_EFFECTS_PSEUDO_CATEGORY:
        _render_status_effects_panel()
        return

    entities = storage.list_entities(category)
    if not entities:
        st.write(L("이 카테고리에 등록된 엔티티가 없습니다."))
        return

    for entity in entities:
        if st.button(_dictionary_label(category, entity), key=f"dict_{entity['id']}"):
            _navigate_to_entity(entity["id"])
            st.rerun()


# ---------------------------------------------------------------------------
# Entity detail — field editor + related-context review (Track A + Track B)
# ---------------------------------------------------------------------------

_EVENT_POINTER_CATEGORIES = ("character", "location", "faction", "artifact", "race")


_RELEVANT_CONTEXT_SHOW_TOP = 3


def _render_relevant_context_section(entity_id: str, field_name: str) -> list:
    """Phase 10 patch 8 — replaces the old "dump every event this entity
    points at" panel (which showed a destroyed_year edit dawnblade's own
    forging event, etc.) with field_update.find_relevant_context: a 1-hop
    walk to *other* entities sharing an event with this one, judged for
    relevance to *this specific edit* by a single batched LLM call.

    Phase 10 patch 11, A: per-item flag checkboxes and top-N/"더보기"
    pagination (mirroring field_update.py's CLI flow, _print_related_context/
    _flag_related_docs) were dropped when this panel was rewritten for the
    1-hop search in patch 8 — restored here. Checkbox state is written
    straight into st.session_state.detail_flag_selection so the save button
    below (_render_save_section) can register flags for whatever's checked
    without this function needing to return anything itself; the return
    value is only for the "관련 기록이 없습니다" caller-side check.

    Phase 10 patch 19: matches are read from st.session_state.
    detail_relevant_matches, computed once by the "필드 값 검토" button
    handler — not recomputed here. This function re-renders on every
    unrelated widget interaction on this screen (a flag checkbox, "더보기"),
    and find_relevant_context's LLM call has no reason to re-fire just
    because Streamlit reran the script with the same entity/field/value.
    """
    st.subheader(L("관련 기록"))
    matches = st.session_state.detail_relevant_matches
    if not matches:
        st.write(L("관련성이 있어 보이는 기록이 없습니다."))
        return matches

    show_all = st.session_state.detail_relevant_show_all
    visible = matches if show_all else matches[:_RELEVANT_CONTEXT_SHOW_TOP]
    remaining = 0 if show_all else len(matches) - len(visible)

    for match in visible:
        st.write(f"**{match.entity_id}**")
        st.caption(match.reason)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.session_state.detail_flag_selection[match.entity_id] = st.checkbox(
                L("플래그"), key=f"flag_{entity_id}_{field_name}_{match.entity_id}"
            )
        with col2:
            if st.button(L("{0} 상세 보기").format(match.entity_id), key=f"relctx_{entity_id}_{field_name}_{match.entity_id}"):
                _navigate_to_entity(match.entity_id)
                st.rerun()

    if remaining:
        if st.button(L("더 보기 ({0}건 더 있음)").format(remaining), key=f"relctx_more_{entity_id}_{field_name}"):
            st.session_state.detail_relevant_show_all = True
            st.rerun()

    return matches


def _render_field_editor_section(category: str, entity_id: str, entity: dict) -> None:
    field_defs = [f for f in schema.get_fields(category) if f["name"] != "event_ids"]
    field_names = [f["name"] for f in field_defs]

    st.subheader(L("필드 수정"))
    selected_field = st.selectbox(L("필드 선택"), field_names, key="detail_field_select")

    if st.session_state.detail_field_name != selected_field:
        _rollback_pending_structured_write()
        st.session_state.detail_field_name = selected_field
        st.session_state.detail_previous_value = entity.get(selected_field)
        st.session_state.detail_searched = False
        st.session_state.detail_conflicts = []
        st.session_state.detail_flag_selection = {}
        st.session_state.detail_relevant_matches = []
        st.session_state.detail_relevant_show_all = False

    field_def = next(f for f in field_defs if f["name"] == selected_field)
    new_value = _render_value_field(
        field_def, entity.get(selected_field), key=f"detail_value_{entity_id}_{selected_field}"
    )

    if st.button(L("필드 값 검토"), key="detail_search"):
        structured = field_update.is_structured_field(category, selected_field)
        if structured:
            storage.save_entity(category, entity_id, {selected_field: new_value})
            st.session_state.detail_conflicts = hard_check.run_hard_checks(category, entity_id)
        else:
            st.session_state.detail_conflicts = []
        st.session_state.detail_new_value = new_value
        st.session_state.detail_searched = True
        st.session_state.detail_flag_selection = {}
        st.session_state.detail_relevant_show_all = False
        # Computed once here (Phase 10 patch 19), not inside the render
        # section below — that section re-renders on every unrelated widget
        # interaction (a flag checkbox, "더보기"), and find_relevant_context
        # is a batched LLM call that has no reason to re-fire just because
        # Streamlit reran the script with the same entity/field/value.
        is_lifecycle = field_def.get("role") in ("lifecycle_start", "lifecycle_end")
        if category in _EVENT_POINTER_CATEGORIES and not is_lifecycle:
            st.session_state.detail_relevant_matches = field_update.find_relevant_context(
                entity_id, selected_field, new_value
            )
        else:
            st.session_state.detail_relevant_matches = []
        st.rerun()

    if not st.session_state.detail_searched:
        return

    conflicts = st.session_state.detail_conflicts
    blocking = [c for c in conflicts if c.severity == "blocking"]
    warnings = [c for c in conflicts if c.severity == "warning"]

    if blocking:
        st.error(L("하드체크 위반으로 저장할 수 없습니다:"))
        for c in blocking:
            st.write(f"- [{c.check_type}] {c.entity_id}: {c.reason}")
    for c in warnings:
        st.warning(f"[{c.check_type}] {c.entity_id}: {c.reason}")
    if not conflicts and field_def.get("role") in ("lifecycle_start", "lifecycle_end"):
        # Phase 10 patch 7 follow-up: a lifecycle field's only meaningful
        # "related event" question is "does this value conflict with a
        # recorded year" — hard_check already answers that above when it
        # fires. When it doesn't, say so explicitly instead of leaving a
        # blank gap where a warning would otherwise have been.
        st.success(L("타임라인 충돌이 감지되지 않았습니다."))

    # The actual "저장" button lives after the relevant-context section
    # below (see _render_save_section) — Phase 10 patch 11, A: flagging a
    # related record and saving the field value must happen together in one
    # click, so the button can't be rendered until the flag checkboxes exist.


def _render_save_section(category: str, entity_id: str, selected_field: str) -> None:
    """The field-editor's actual commit step (Phase 10 patch 11, A) — split
    out from _render_field_editor_section so it can render after the
    relevant-context section's flag checkboxes exist, and register both in
    one click: the new field value, any items checked "플래그" above (via
    flags.add_flag, same flagged_from convention field_update.py's CLI flow
    uses), and — the part that had gone missing entirely, not just the
    checkboxes — auto-clearing whatever flags were already sitting against
    this entity (flags.clear_flags_for_entity), same as the CLI's
    update_field_flow does on every successful save."""
    conflicts = st.session_state.detail_conflicts
    blocking = [c for c in conflicts if c.severity == "blocking"]
    warnings = [c for c in conflicts if c.severity == "warning"]

    if st.button(L("저장"), key="detail_save", disabled=bool(blocking)):
        if not field_update.is_structured_field(category, selected_field):
            storage.save_entity(category, entity_id, {selected_field: st.session_state.detail_new_value})
        if any(c.check_type == "lifespan" for c in warnings):
            storage.save_entity("character", entity_id, {"lifespan_check_ack": True})

        flagged_from = f"{entity_id}의 {selected_field} 수정 중 발견"
        flagged_count = 0
        for flagged_entity_id, checked in st.session_state.detail_flag_selection.items():
            if checked:
                flags.add_flag(flagged_entity_id, flagged_from)
                flagged_count += 1

        flags.clear_flags_for_entity(entity_id)

        message = L("저장 완료.")
        if flagged_count:
            message += L(" {0}건 플래그 등록.").format(flagged_count)
        st.success(message)

        st.session_state.detail_searched = False
        st.session_state.detail_conflicts = []
        st.session_state.detail_flag_selection = {}
        st.session_state.detail_relevant_matches = []
        st.session_state.detail_relevant_show_all = False
        st.rerun()


def _render_delete_entity_section(category: str, entity_id: str) -> None:
    st.subheader(L("엔티티 삭제"))
    if not st.session_state.detail_confirm_delete:
        if st.button(L("이 엔티티 삭제"), key="detail_delete_start"):
            st.session_state.detail_confirm_delete = True
            st.rerun()
        return

    events = deletion.request_entity_deletion(entity_id)
    if events:
        st.write(L("이 엔티티가 관여한 이벤트 {0}건도 함께 정리됩니다 (다른 엔티티가 "
                    "관여하지 않은 이벤트는 삭제, 관여했다면 이 엔티티의 포인터만 제거):").format(len(events)))
        for record in events:
            st.write(f"- {record['id']}: {record.get('notes') or ''}")

    col1, col2 = st.columns(2)
    if col1.button(L("그대로 삭제 진행"), key="detail_delete_confirm"):
        result = deletion.delete_entity(entity_id, category)
        st.session_state.detail_confirm_delete = False
        _navigate_to_entity(None)
        message = L("{0} 삭제 완료.").format(result.deleted_id)
        if result.deleted_events:
            message += L(" 함께 삭제된 이벤트: {0}.").format(', '.join(result.deleted_events))
        if result.affected_entities:
            message += L(" 포인터가 갱신된 엔티티: {0}.").format(', '.join(result.affected_entities))
        st.success(message)
        st.rerun()
    if col2.button(L("취소 (유지)"), key="detail_delete_cancel"):
        st.session_state.detail_confirm_delete = False
        st.rerun()


def _participant_options() -> list:
    """Every entity that could plausibly appear as an event participant —
    every name-bearing, event-pointer category. `system` deliberately isn't
    here: it has neither event_ids nor any narrative role to play in a
    timeline record."""
    options = []
    for category in _NAME_BEARING_CATEGORIES:
        for e in storage.list_entities(category):
            options.append(e["id"])
    return options


def _participant_label(entity_id: str) -> str:
    category = schema.category_from_id(entity_id)
    if category is None:
        return entity_id
    entity = storage.get_entity(category, entity_id) or {}
    return f'{entity.get("name") or entity_id} ({category})'


def _render_timeline_detail(entity_id: str, entity: dict) -> None:
    """Phase 10 patch 8, section 4 (rolled back by patch 11, A) — "editing"
    an event used to be a delete-and-recreate, on the reasoning that a
    partial patch couldn't fix the event's own id (a content-derived slug).
    That churned the id on every edit, which broke anything that had
    captured the old id directly — most concretely, a flag on this event
    (flags.py keys off entity_id) went stale and orphaned the moment the
    event got "fixed," the opposite of what flagging something for a later
    look is for. The id is just an internal identifier, same as any other
    entity's — it doesn't need to keep matching the event's current content,
    any more than a character's id needs to keep matching their current
    name after a rename. So this edits the row in place, same id throughout:
    `storage.save_entity`/`save_to_chroma` overwrite the existing record,
    and only the *participant* pointers get reconciled (diffed against who
    was already pointing at this event_id before the edit) since who's
    involved can genuinely change — everyone else's pointer to this id
    stays untouched and valid. Verification is still hard_check only —
    rag_check's LLM judgments are Step 4's save-time contradiction checks
    for *new* input, redundant (and potentially confusing) for a human who
    already reviewed this exact record via the relevant-context search and
    chose to fix it directly. No gating on how you got here — reachable
    from the dictionary's "timeline" category or from a relevant-context
    match, whichever came first."""
    is_point = entity.get("year") is not None or entity.get("entity") is None
    candidates = _participant_options()
    labels_by_id = {eid: _participant_label(eid) for eid in candidates}

    if is_point:
        previous_participants = [eid for _cat, eid in storage.find_entities_referencing_event(entity_id)]
    else:
        previous_participants = [p for p in (entity.get("entity"), entity.get("target")) if p]

    st.subheader(L("이벤트 수정"))
    if is_point:
        new_year = st.number_input(L("연도"), value=entity.get("year"), step=1, key=f"tl_year_{entity_id}")
        loc_options = [(f'{e.get("name") or e["id"]}', e["id"]) for e in storage.list_entities("location")]
        loc_ids = [eid for _label, eid in loc_options]
        loc_labels = [L("(없음)")] + [label for label, _eid in loc_options]
        loc_index = loc_ids.index(entity.get("location")) + 1 if entity.get("location") in loc_ids else 0
        loc_choice = st.selectbox(L("장소"), loc_labels, index=loc_index, key=f"tl_loc_{entity_id}")
        new_location = None if loc_choice == L("(없음)") else loc_ids[loc_labels.index(loc_choice) - 1]

        selected_participants = st.multiselect(
            L("참가자"), candidates, default=[p for p in previous_participants if p in candidates],
            format_func=lambda eid: labels_by_id.get(eid, eid), key=f"tl_participants_{entity_id}",
        )
    else:
        new_start = st.number_input(L("시작 연도"), value=entity.get("start_year"), step=1, key=f"tl_start_{entity_id}")
        new_end = st.number_input(L("종료 연도 (비워두면 현재도 진행 중)"), value=entity.get("end_year"), step=1, key=f"tl_end_{entity_id}")
        new_predicate = st.text_input(L("predicate (상태/관계 이름)"), value=entity.get("predicate") or "", key=f"tl_pred_{entity_id}")
        entity_index = candidates.index(entity.get("entity")) if entity.get("entity") in candidates else 0
        new_entity = st.selectbox(
            L("주체 (entity)"), candidates, index=entity_index,
            format_func=lambda eid: labels_by_id.get(eid, eid), key=f"tl_entity_{entity_id}",
        ) if candidates else None
        target_labels = [L("(없음)")] + [labels_by_id[eid] for eid in candidates]
        target_index = candidates.index(entity.get("target")) + 1 if entity.get("target") in candidates else 0
        target_choice = st.selectbox(L("대상 (target, 관계형일 때만)"), target_labels, index=target_index, key=f"tl_target_{entity_id}")
        new_target = None if target_choice == L("(없음)") else candidates[target_labels.index(target_choice) - 1]
        selected_participants = [p for p in (new_entity, new_target) if p]

    new_notes = st.text_area(L("비고"), value=entity.get("notes") or "", key=f"tl_notes_{entity_id}")

    review_key = f"tl_reviewed_{entity_id}"
    pending_key = f"tl_pending_{entity_id}"

    if st.button(L("변경사항 검토"), key=f"tl_review_{entity_id}"):
        if is_point:
            fields = {"year": int(new_year) if new_year is not None else None, "location": new_location, "notes": new_notes}
        else:
            fields = {
                "entity": new_entity, "predicate": new_predicate or None, "target": new_target,
                "start_year": int(new_start) if new_start is not None else None,
                "end_year": int(new_end) if new_end is not None else None,
                "notes": new_notes,
            }
        st.session_state[pending_key] = {"fields": fields, "participants": selected_participants}
        st.session_state[review_key] = True
        st.rerun()

    st.subheader(L("이벤트 삭제"))
    if st.button(L("이 이벤트 삭제"), key=f"tl_delete_{entity_id}"):
        result = deletion.delete_event(entity_id)
        message = L("{0} 삭제 완료.").format(result.deleted_id)
        if result.affected_entities:
            message += L(" 포인터가 제거된 엔티티: {0}").format(", ".join(result.affected_entities))
        st.success(message)
        _navigate_to_entity(None)
        st.rerun()

    if not st.session_state.get(review_key):
        return

    pending = st.session_state[pending_key]
    conflicts = []
    extra_years = [
        y for y in (
            pending["fields"].get("year"),
            pending["fields"].get("start_year"),
            pending["fields"].get("end_year"),
        )
        if y is not None
    ]
    for participant in pending["participants"]:
        category = schema.category_from_id(participant)
        if category is not None:
            conflicts.extend(hard_check.run_hard_checks(category, participant, extra_years=extra_years))

    blocking = [c for c in conflicts if c.severity == "blocking"]
    warnings = [c for c in conflicts if c.severity == "warning"]
    if blocking:
        st.error(L("하드체크 위반으로 저장할 수 없습니다:"))
        for c in blocking:
            st.write(f"- [{c.check_type}] {c.entity_id}: {c.reason}")
    for c in warnings:
        st.warning(f"[{c.check_type}] {c.entity_id}: {c.reason}")
    if not conflicts:
        st.success(L("하드체크 충돌이 감지되지 않았습니다."))

    if st.button(L("저장"), key=f"tl_save_{entity_id}", disabled=bool(blocking)):
        storage.save_entity("timeline", entity_id, pending["fields"])
        storage.save_to_chroma(entity_id, pending["fields"].get("notes") or "", {"category": "timeline"})

        new_participants = set(pending["participants"])
        for participant in set(previous_participants) - new_participants:
            storage.remove_event_pointer(participant, entity_id)
        for participant in new_participants - set(previous_participants):
            storage.add_event_pointer(participant, entity_id)

        flags.clear_flags_for_entity(entity_id)

        st.session_state.pop(review_key, None)
        st.session_state.pop(pending_key, None)
        st.success(L("이벤트가 갱신되었습니다."))
        st.rerun()


def _render_entity_timeline(entity_id: str) -> None:
    """Phase 10 patch 17, B — this entity's own events on a year axis.
    Point events render as markers on a shared "사건" row; duration events
    each get their own horizontal bar row. No LLM call anywhere — labels
    are either a plain truncation of `notes` (point) or `predicate` + the
    target's name (duration), both already computed by
    visualization.build_timeline.

    Point events deliberately stay on ONE shared lane rather than each
    getting its own row the way duration events do — a duration event is
    inherently rare (one per status/relationship an entity ever has), but
    a busy character could easily rack up dozens of point events, and a
    row-per-event layout would make the whole chart's height scale with
    that count. The tradeoff: point events show detail on hover only, not
    as permanent floating text (which would collide the moment two events
    landed close together on the year axis, or land on top of each other
    entirely when they share the exact same year — see the jitter
    below)."""
    import plotly.graph_objects as go

    entries = visualization.build_timeline(entity_id)
    if not entries:
        st.write(L("이 엔티티와 관련된 사건이 없습니다."))
        return

    point_entries = [e for e in entries if e.kind == "point"]
    duration_entries = [e for e in entries if e.kind == "duration"]

    # Phase 10 patch 17 follow-up: an "ongoing" (end_year=None) bar needs
    # something to visually stop at. entity_id's own lifecycle_end
    # (death/destroyed/disbanded — whichever role its category defines)
    # is a real, known fact when set, so a bar gets capped there and shown
    # as genuinely ended. Otherwise there's no known endpoint, so it falls
    # back to a "cutoff" guessed from entity_id's own chronologically last
    # event (its year if that event is a point; its start_year + a small
    # buffer if a duration — see resolve_timeline_reference's docstring
    # for why the buffer exists) and stays visually distinct: lighter
    # color, "(진행중)" label, dashed reference line.
    ref = visualization.resolve_timeline_reference(entity_id)
    open_bound = ref["end"] or ref["cutoff"]

    fig = go.Figure()
    for e in duration_entries:
        # `.x` (plot position) caps the bar's geometry; `.year` (the real
        # fact) is what any text shown to a person should say instead —
        # for a guessed cutoff these two differ on purpose (see
        # visualization.ReferenceLine), so the bar can render with visible
        # width without the hover text claiming a fictional later year.
        bound_x = open_bound.x if open_bound else e.start_year
        bound_year = open_bound.year if open_bound else e.start_year
        start = e.start_year if e.start_year is not None else bound_x
        if e.end_year is not None:
            end = e.end_year
            suffix = ""
            color = "#2a78d6"
            span_label = f"{start}~{end}"
        elif ref["end"] is not None:
            end = max(bound_x, start)
            suffix = L(" (대상 소멸로 종료)")
            color = "#2a78d6"
            span_label = f"{start}~{bound_year}" + L(" (대상 소멸로 종료)")
        else:
            end = max(bound_x, start)
            suffix = L(" (진행중)")
            color = "#9ec5f4"
            span_label = f"{start}~" + L(" (마지막 기록: {0}년)").format(bound_year)
        fig.add_trace(
            go.Bar(
                x=[max(end - start, 0.5)],
                y=[e.label + suffix],
                base=[start],
                orientation="h",
                customdata=[e.event_id],
                hovertext=[span_label],
                hoverinfo="text",
                marker_color=color,
                showlegend=False,
            )
        )

    # Populated below when there are point events; kept accessible after
    # the chart is drawn so a click on a multi-event year's marker (no
    # single unambiguous target — see below) can still offer a picker
    # instead of just doing nothing.
    by_year: dict = {}
    if point_entries:
        # Jittering apart same-year markers (an earlier version of this
        # fix) turned out to shrink to invisibility on any chart spanning
        # more than a few years — a 0.15-year nudge is imperceptible next
        # to a 50+ year axis. Grouping by year instead sidesteps the whole
        # "how big a nudge is enough" problem: one marker per distinct
        # year, and every event that year lists in the hover popup. A
        # single-event year still click-navigates straight to that event;
        # a multi-event year has no single unambiguous click target, so a
        # click there offers a picker instead (see below the chart).
        for e in point_entries:
            by_year.setdefault(e.year, []).append(e)

        years_sorted = sorted(by_year)
        hover_texts = []
        customdata = []
        for year in years_sorted:
            group = by_year[year]
            if len(group) == 1:
                hover_texts.append(L("{0}년, 사건: {1}").format(year, group[0].label))
                customdata.append(group[0].event_id)
            else:
                lines = "<br>".join(f"• {e.label}" for e in group)
                hover_texts.append(L("{0}년, {1}개 사건:").format(year, len(group)) + f"<br>{lines}")
                customdata.append(None)

        fig.add_trace(
            go.Scatter(
                x=years_sorted,
                y=[L("사건")] * len(years_sorted),
                mode="markers",
                customdata=customdata,
                hovertext=hover_texts,
                hoverinfo="text",
                marker=dict(size=12, color="#E45756"),
                showlegend=False,
            )
        )

    for line, dash in ((ref["start"], "dot"), (ref["end"], "dot"), (ref["cutoff"], "dash")):
        if line is not None:
            # visualization.py deliberately has no i18n import of its own
            # (kept GUI-framework-free by design) — L() here is a no-op for
            # a real schema field name (birth_year, founded_year, ...),
            # since those never match a dictionary entry, and only
            # translates the two UI-generated guess labels ("마지막 이벤트",
            # "첫 기록") that actually are.
            fig.add_vline(
                x=line.x, line_dash=dash, line_color="#898781",
                annotation_text=f"{L(line.label)}: {line.year}", annotation_position="top",
                annotation_textangle=0,
            )

    fig.update_layout(
        xaxis_title=L("연도"),
        height=max(300, 70 * (len(duration_entries) + 1)),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    event = st.plotly_chart(fig, on_select="rerun", key=f"timeline_chart_{entity_id}")
    points = (event.get("selection") or {}).get("points") or []
    # customdata was set here as a flat 1D array (one scalar per point), so
    # plotly reports it back as that scalar directly — NOT wrapped in an
    # extra list the way Streamlit's own docs example shows (that example
    # used px.scatter's hover_data mechanism, which produces a 2D
    # customdata array; ours doesn't). Indexing with an extra `[0]` here
    # was silently slicing the first *character* off the event_id string
    # instead of reading it — confirmed from the exact reported symptom
    # ("Unknown entity_id: e", the first letter of "event_...").
    if points:
        clicked_id = points[0].get("customdata")
        if clicked_id:
            _navigate_to_entity(clicked_id)
            st.rerun()
        elif points[0].get("x") in by_year:
            # A multi-event year's marker carries no customdata (no single
            # unambiguous click target) — offer a picker instead of just
            # silently doing nothing, using the clicked point's own x
            # (the year) to look the group back up.
            group = by_year[points[0]["x"]]
            st.info(L("{0}년에 사건이 {1}개 있습니다. 이동할 사건을 선택하세요:").format(points[0]['x'], len(group)))
            for ge in group:
                if st.button(ge.label, key=f"timeline_pick_{entity_id}_{ge.event_id}"):
                    _navigate_to_entity(ge.event_id)
                    st.rerun()


def _render_relationship_graph(entity_id: str) -> None:
    """Phase 10 patch 17, C — hub-and-spoke: entity_id at the center, every
    1-hop neighbor (any category, point or duration events alike) around
    it, edge thickness = shared event count. Category checkboxes and the
    minimum-connections slider are both computed from live data, never a
    hardcoded list/range."""
    from streamlit_agraph import Config, Edge, Node, agraph

    category = schema.category_from_id(entity_id)
    center_entity = storage.get_entity(category, entity_id) if category else None
    all_time_weights = visualization.compute_neighbor_weights(entity_id)

    if not all_time_weights:
        st.write(L("1-hop으로 연결된 엔티티가 없습니다."))
        return

    # Year slider, bounds from the same start/end (or first/last-record
    # guess) reference resolution the Timeline tab already uses — every
    # relationship shown is filtered to what actually exists as of this
    # year (see visualization._event_active_at), matching the project's
    # own "no fixed now, everything computed from the timeline" stance
    # rather than the flat all-time graph this used to render.
    ref = visualization.resolve_timeline_reference(entity_id)
    year_min = ref["start"].year
    year_max = (ref["end"] or ref["cutoff"]).year

    # Both sliders share one row — year on the left, the count filter on
    # the right — rather than stacking, since they're two independent
    # filters over the same graph, not a sequence of steps.
    year_col, count_col = st.columns(2)
    with year_col:
        if year_max > year_min:
            as_of_year = st.slider(
                L("기준 연도"), min_value=year_min, max_value=year_max, value=year_max,
                key=f"graph_year_{entity_id}",
            )
        else:
            as_of_year = year_max

    weights = visualization.compute_neighbor_weights(entity_id, as_of_year=as_of_year)
    if not weights:
        st.write(L("선택한 연도 기준으로 존재하는 관계가 없습니다."))
        return

    # st.slider requires min_value < max_value — every neighbor sharing
    # exactly 1 event (a common case, not just the empty-weights one
    # already handled above) would otherwise make max_weight == 1 and
    # crash the widget. A 1~1 range has nothing meaningful to filter
    # anyway, so just skip the slider and show everyone in that case.
    max_weight = max(weights.values())
    with count_col:
        if max_weight > 1:
            min_weight = st.slider(
                L("최소 연결 횟수"), min_value=1, max_value=max_weight, value=1,
                key=f"graph_minweight_{entity_id}",
            )
        else:
            min_weight = 1

    st.write(L("**카테고리 필터**"))
    categories = visualization.filterable_categories()
    selected_categories = set()
    if categories:
        cols = st.columns(len(categories))
        for col, cat in zip(cols, categories):
            with col:
                if st.checkbox(cat, value=True, key=f"graph_cat_{entity_id}_{cat}"):
                    selected_categories.add(cat)

    nodes_data, edges_data = visualization.build_relationship_graph(
        entity_id, weights, selected_categories, min_weight
    )
    if not nodes_data:
        st.write(L("조건에 맞는 연결된 엔티티가 없습니다."))
        return

    # Legend first (dataviz skill non-negotiable: identity is never
    # color-alone once there's more than one series) — only for the
    # categories actually present among the nodes shown, not every
    # filterable category, so it doesn't advertise colors nothing on
    # screen is using.
    shown_categories = sorted({n.category for n in nodes_data})
    if shown_categories:
        legend_cols = st.columns(len(shown_categories))
        for col, cat in zip(legend_cols, shown_categories):
            with col:
                st.markdown(
                    f'<span style="color:{visualization.category_color(cat)}">●</span> {cat}',
                    unsafe_allow_html=True,
                )

    edge_weights = [e.weight for e in edges_data]
    min_w, max_w = min(edge_weights), max(edge_weights)

    # vis-network's default node font color is dark, close to unreadable
    # against Streamlit's dark theme background — force a light one instead.
    _node_font = {"color": "#f5f5f5"}
    agraph_nodes = [
        Node(
            id=entity_id, label=(center_entity or {}).get("name") or entity_id, size=30,
            color=visualization.CENTER_NODE_COLOR, font=_node_font,
        )
    ]
    agraph_nodes += [
        Node(id=n.entity_id, label=n.label, size=20, color=visualization.category_color(n.category), font=_node_font)
        for n in nodes_data
    ]
    agraph_edges = [
        Edge(source=entity_id, target=e.other_id, width=e.weight, color=visualization.edge_color(e.weight, min_w, max_w))
        for e in edges_data
    ]

    config = Config(width=700, height=500, directed=False, physics=True)
    clicked = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
    if clicked and clicked != entity_id:
        _navigate_to_entity(clicked)
        st.rerun()


def render_entity_detail(entity_id: str) -> None:
    category = schema.category_from_id(entity_id)
    if category is None:
        st.error(L("알 수 없는 entity_id입니다: {0}").format(entity_id))
        return
    entity = storage.get_entity(category, entity_id)
    if entity is None:
        st.error(L("존재하지 않는 엔티티입니다: {0}").format(entity_id))
        return

    st.header(f"{entity.get('name') or entity_id} ({entity_id})")
    if st.button(L("← 목록으로"), key="detail_back"):
        _navigate_to_entity(None)
        st.rerun()

    if category == "timeline":
        # Phase 10 patch 8, section 4: an event has no fields that can be
        # patched one at a time without invalidating its own content-derived
        # id, so it gets a dedicated form instead of the generic field
        # editor — see _render_timeline_detail. Phase 10 patch 17's
        # visualization tabs are deliberately not offered here either — a
        # raw timeline record has no timeline/relationship graph of its
        # own to show.
        st.subheader(L("현재 필드 값"))
        _render_entity_fields(category, entity_id, entity)
        _render_timeline_participants(entity_id, entity)
        _render_timeline_detail(entity_id, entity)
        return

    info_tab, viz_tab = st.tabs([L("정보/수정"), L("시각화")])

    with info_tab:
        st.subheader(L("현재 필드 값"))
        _render_entity_fields(category, entity_id, entity)
        _render_field_editor_section(category, entity_id, entity)

        # The relevance search only runs once a field edit is actually
        # attempted this session (detail_searched, set by "필드 값 검토" below),
        # and never for lifecycle fields (birth_year/death_year/founded_year/
        # ...) — the only thing "related" could mean there is "does this year
        # conflict with a recorded event", and hard_check already answers that
        # directly above (see _render_field_editor_section's success/warning
        # messages).
        selected_field_def = next(
            (f for f in schema.get_fields(category) if f["name"] == st.session_state.get("detail_field_name")),
            None,
        )
        is_lifecycle_field = bool(selected_field_def and selected_field_def.get("role") in ("lifecycle_start", "lifecycle_end"))
        if st.session_state.get("detail_searched"):
            if category in _EVENT_POINTER_CATEGORIES and not is_lifecycle_field:
                _render_relevant_context_section(entity_id, st.session_state.detail_field_name)
            _render_save_section(category, entity_id, st.session_state.detail_field_name)
        _render_delete_entity_section(category, entity_id)

    with viz_tab:
        timeline_tab, graph_tab = st.tabs([L("타임라인"), L("관계도")])
        with timeline_tab:
            _render_entity_timeline(entity_id)
        with graph_tab:
            _render_relationship_graph(entity_id)


# ---------------------------------------------------------------------------
# Top-level layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Lore Builder", layout="wide")
    _init_session_state()

    st.sidebar.title("Lore Builder")
    st.sidebar.selectbox(
        "Language / 언어",
        ["ko", "en"],
        format_func=lambda code: "한국어" if code == "ko" else "English",
        key="interface_language",
    )
    render_sidebar_search()
    st.sidebar.divider()

    # format_func=L would read interface_language via st.session_state on
    # every invocation, including ones Streamlit's own testing harness makes
    # outside a live script run (no ScriptRunContext there) — that path sees
    # stale/default state and can desync the widget's displayed vs. stored
    # options. Building the translated list up front and reverse-mapping
    # keeps format_func itself state-free, sidestepping that entirely (same
    # pattern as the status-effect type selectbox below).
    _mode_options = ["채팅", "딕셔너리"]
    _mode_display = [L(m) for m in _mode_options]
    mode_choice = st.sidebar.radio(L("모드"), _mode_display, key="mode")
    mode = _mode_options[_mode_display.index(mode_choice)]

    # Phase 9 patch E: clicking a different mode tab must win over "an
    # entity detail screen happens to be open" — previously the
    # selected_entity check below ran unconditionally every rerun and never
    # noticed the mode had changed, so switching tabs while viewing an
    # entity did nothing until you clicked "← 목록으로" first.
    if st.session_state._last_mode is not None and st.session_state._last_mode != mode:
        _navigate_to_entity(None)
    st.session_state._last_mode = mode

    with st.sidebar.expander(L("🚩 플래그 확인")):
        deduped = flags.list_flags_deduped()
        if not deduped:
            st.write(L("플래그된 항목이 없습니다."))
        for flag in deduped:
            # No GUI path ever collects a reason when flagging (add_flag is
            # always called with just entity_id/flagged_from), so this was
            # unconditionally showing a "(사유 없음)" placeholder — pure noise
            # with no way to ever make it say anything else. Only show the
            # reason when one actually exists (e.g. flags added some other
            # way with a real reason).
            label = f"{flag.entity_id} — {flag.reason}" if flag.reason else flag.entity_id
            if st.button(label, key=f"flagnav_{flag.id}"):
                _navigate_to_entity(flag.entity_id)
                st.rerun()

    if st.session_state.selected_entity:
        render_entity_detail(st.session_state.selected_entity)
        return

    if mode == "채팅":
        render_chat_mode()
    elif mode == "딕셔너리":
        render_dictionary_mode()
    else:
        st.info(L("시각화 모드는 곧 추가될 예정입니다."))


if __name__ == "__main__":
    main()
