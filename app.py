"""Lore Builder GUI — Phase 9 (Streamlit).

Runs in-process, calling pipeline_session.py/field_update.py/flags.py
directly — no separate API server. Sidebar has a permanent search box at
the top (not a mode — it's live no matter which mode is active) and 3
mode tabs below it (chat / dictionary / visualization-placeholder).
"review pending" is deliberately NOT a mode either — it's just what the
entity-detail screen becomes once you pick a field to edit, reached from
either the search box or the dictionary.

Widget polish (badges, styling) is explicitly out of scope for this phase —
the goal is being able to repeat every CLI test scenario through the GUI.
"""

import streamlit as st

from src import field_update, flags, hard_check, pipeline_session, schema, storage

_NAME_BEARING_CATEGORIES = ("character", "location", "faction", "artifact", "race")


# ---------------------------------------------------------------------------
# Session-state setup
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "session": None,
        "selected_entity": None,
        "detail_field_name": None,
        "detail_previous_value": None,
        "detail_searched": False,
        "detail_conflicts": [],
        "detail_related_docs": [],
        "detail_new_value": None,
        "detail_flag_selection": {},
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
    st.session_state.detail_related_docs = []
    st.session_state.detail_flag_selection = {}
    st.session_state.selected_entity = entity_id


# ---------------------------------------------------------------------------
# Chat mode — renders pipeline_session's 7 decision types
# ---------------------------------------------------------------------------

def _resume(session, response) -> None:
    st.session_state.session = pipeline_session.resume_session(session.session_id, response)
    st.rerun()


def _describe_result(result: dict) -> str:
    status = result.get("status")
    if status == "error":
        return f"입력 오류: {result['message']}"
    if status == "rejected" and result.get("stage") == "hard_check":
        lines = ["하드체크 결과에 따라 저장이 중단되었습니다."]
        for c in result.get("conflicts", []):
            if c.severity == "blocking":
                lines.append(f"- [{c.check_type}] {c.entity_id}: {c.reason}")
        return "\n".join(lines)
    if status == "rejected" and result.get("stage") == "rag_check":
        return "RAG 검증 결과에 따라 저장이 중단되었습니다."
    if status == "no_changes":
        return "승인된 변경사항이 없어 저장할 내용이 없습니다."
    if status == "saved":
        applied = result.get("applied", [])
        names = ", ".join(
            f"{c.entity_id}(갱신)" if c.action == "update" else c.entity_id for c in applied
        )
        return f"저장 완료: {names}"
    return "완료되었습니다."


def _render_entity_candidates(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.write(f'"{payload["tag"]}" 후보를 선택하세요:')
    for i, candidate_id in enumerate(payload["candidates"]):
        if st.button(candidate_id, key=f"{key_prefix}_cand_{i}"):
            _resume(session, candidate_id)
    if payload.get("allow_create") and st.button("새로 작성", key=f"{key_prefix}_create"):
        _resume(session, pipeline_session.CREATE_NEW)


def _render_entity_name(session, decision, key_prefix) -> None:
    payload = decision.payload
    value = st.text_input(
        f'"{payload["tag"]}" 이름', value=payload["default"], key=f"{key_prefix}_name"
    )
    if st.button("확인", key=f"{key_prefix}_name_confirm"):
        _resume(session, value)


def _render_entity_terminal_status(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.write(
        f"[{payload['tag']}]가 이 사건({payload['year']}년)으로 사망(또는 활동 종료)한 것으로 "
        f"추정됩니다. death_year={payload['year']}로 저장할까요?"
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("예", key=f"{key_prefix}_yes"):
        _resume(session, "예")
    if col2.button("아니오", key=f"{key_prefix}_no"):
        _resume(session, "아니오")
    if col3.button("수정", key=f"{key_prefix}_edit_toggle"):
        st.session_state[f"{key_prefix}_editing"] = True
    if st.session_state.get(f"{key_prefix}_editing"):
        new_year = st.number_input(
            "새로운 death_year 값", value=payload["year"], step=1, key=f"{key_prefix}_year"
        )
        if st.button("death_year로 저장", key=f"{key_prefix}_edit_confirm"):
            _resume(session, {"수정": {"death_year": int(new_year)}})


def _render_entity_required_field(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.write(f"[{payload['category']}] 필수 필드를 입력하세요:")
    values = {}
    for f in payload["fields"]:
        values[f["name"]] = st.text_input(f["name"], key=f"{key_prefix}_{f['name']}")
    if st.button("저장", key=f"{key_prefix}_submit"):
        _resume(session, values)


def _render_hard_check_warning(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.warning(f"[{payload['entity_id']}] {payload['reason']}")
    col1, col2, col3 = st.columns(3)
    if col1.button("그래도 저장", key=f"{key_prefix}_accept"):
        _resume(session, "그래도 저장")
    if col2.button("수정", key=f"{key_prefix}_revise"):
        _resume(session, "수정")
    if col3.button("취소", key=f"{key_prefix}_cancel"):
        _resume(session, "취소")


def _render_rag_judgment(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.warning(f"[{payload['judgment_type']}] {payload['reason']}")
    col1, col2, col3 = st.columns(3)
    if col1.button("그래도 저장", key=f"{key_prefix}_accept"):
        _resume(session, "그래도 저장")
    if col2.button("수정", key=f"{key_prefix}_revise"):
        _resume(session, "수정")
    if col3.button("취소", key=f"{key_prefix}_cancel"):
        _resume(session, "취소")


def _render_diff_item(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.write(f"**{payload['action'].upper()} {payload['category']}**: {payload['entity_id']}")
    st.caption(f"근거: {payload['reason']}")
    st.json(payload["fields"])
    col1, col2 = st.columns(2)
    if col1.button("승인", key=f"{key_prefix}_approve"):
        _resume(session, True)
    if col2.button("거부", key=f"{key_prefix}_reject"):
        _resume(session, False)


_DECISION_RENDERERS = {
    "entity_candidates": _render_entity_candidates,
    "entity_name": _render_entity_name,
    "entity_terminal_status": _render_entity_terminal_status,
    "entity_required_field": _render_entity_required_field,
    "hard_check_warning": _render_hard_check_warning,
    "rag_judgment": _render_rag_judgment,
    "diff_item": _render_diff_item,
}


def render_chat_mode() -> None:
    st.header("채팅")

    text = st.chat_input("사건을 입력하세요")
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
# Sidebar search — always visible, not a mode
# ---------------------------------------------------------------------------

def render_sidebar_search() -> None:
    """Lives at the top of the sidebar regardless of which mode tab is
    active — searching and switching modes are independent actions, not
    alternatives, so this was pulled out of the mode dispatch entirely."""
    query = st.sidebar.text_input("🔍 검색 (이름 일부)", key="global_search")
    if not query:
        return

    results = []
    for category in _NAME_BEARING_CATEGORIES:
        for entity in storage.list_entities(category):
            if query in (entity.get("name") or ""):
                results.append((category, entity))

    if not results:
        st.sidebar.caption("일치하는 엔티티가 없습니다.")
        return

    for category, entity in results:
        label = f"[{category}] {entity['name']} ({entity['id']})"
        if st.sidebar.button(label, key=f"search_{entity['id']}"):
            _navigate_to_entity(entity["id"])
            st.rerun()


def render_dictionary_mode() -> None:
    st.header("딕셔너리")
    categories = schema.list_categories()
    category = st.selectbox("카테고리", categories, key="dict_category")

    entities = storage.list_entities(category)
    if not entities:
        st.write("이 카테고리에 등록된 엔티티가 없습니다.")
        return

    for entity in entities:
        label = entity.get("name") or entity["id"]
        if st.button(f"{label} ({entity['id']})", key=f"dict_{entity['id']}"):
            _navigate_to_entity(entity["id"])
            st.rerun()


# ---------------------------------------------------------------------------
# Entity detail — field editor + related-context review (Track A + Track B)
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
            options.append((f'{e["id"]} ({e.get("year", "?")}년)', e["id"]))
        return options

    entities = storage.list_entities(ref_category)
    if ref_category == "timeline":
        return [(f'{e["id"]} ({e.get("year", "?")}년)', e["id"]) for e in entities]
    return [(f'{e.get("name") or e["id"]} ({e["id"]})', e["id"]) for e in entities]


def _render_value_field(field_def: dict, current_value, key: str):
    field_type = field_def["type"]
    name = field_def["name"]

    if field_type == "reference":
        options = _reference_options(field_def.get("ref_category"))
        labels = [label for label, _id in options]
        ids = [entity_id for _label, entity_id in options]
        index = ids.index(current_value) + 1 if current_value in ids else 0
        choice = st.selectbox(name, ["(비어있음)"] + labels, index=index, key=key)
        return None if choice == "(비어있음)" else ids[labels.index(choice)]

    if field_type == "enum":
        options = field_def.get("options", [])
        index = options.index(current_value) + 1 if current_value in options else 0
        choice = st.selectbox(name, ["(비어있음)"] + options, index=index, key=key)
        return None if choice == "(비어있음)" else choice

    if field_type == "boolean":
        return st.checkbox(name, value=bool(current_value), key=key)

    if field_type == "integer":
        return int(st.number_input(name, value=int(current_value or 0), step=1, key=key))

    if field_type == "list":
        raw = st.text_input(name, value=", ".join(current_value or []), key=key)
        return [v.strip() for v in raw.split(",") if v.strip()]

    return st.text_input(name, value=current_value or "", key=key)


def render_entity_detail(entity_id: str) -> None:
    category = schema.category_from_id(entity_id)
    if category is None:
        st.error(f"알 수 없는 entity_id입니다: {entity_id}")
        return
    entity = storage.get_entity(category, entity_id)
    if entity is None:
        st.error(f"존재하지 않는 엔티티입니다: {entity_id}")
        return

    st.header(f"{entity.get('name') or entity_id} ({entity_id})")
    if st.button("← 목록으로", key="detail_back"):
        _navigate_to_entity(None)
        st.rerun()

    st.subheader("현재 필드 값")
    st.json(entity)

    field_defs = schema.get_fields(category)
    field_names = [f["name"] for f in field_defs]

    st.subheader("필드 수정")
    selected_field = st.selectbox("필드 선택", field_names, key="detail_field_select")

    if st.session_state.detail_field_name != selected_field:
        _rollback_pending_structured_write()
        st.session_state.detail_field_name = selected_field
        st.session_state.detail_previous_value = entity.get(selected_field)
        st.session_state.detail_searched = False
        st.session_state.detail_conflicts = []
        st.session_state.detail_related_docs = []
        st.session_state.detail_flag_selection = {}

    field_def = next(f for f in field_defs if f["name"] == selected_field)
    new_value = _render_value_field(
        field_def, entity.get(selected_field), key=f"detail_value_{entity_id}_{selected_field}"
    )

    if st.button("관련 기록 검색", key="detail_search"):
        structured = field_update.is_structured_field(category, selected_field)
        if structured:
            storage.save_entity(category, entity_id, {selected_field: new_value})
            st.session_state.detail_conflicts = hard_check.run_hard_checks(category, entity_id)
        else:
            st.session_state.detail_conflicts = []
        st.session_state.detail_related_docs = field_update.find_related_context(
            entity_id, selected_field, new_value
        )
        st.session_state.detail_new_value = new_value
        st.session_state.detail_searched = True
        st.rerun()

    if not st.session_state.detail_searched:
        return

    conflicts = st.session_state.detail_conflicts
    blocking = [c for c in conflicts if c.severity == "blocking"]
    warnings = [c for c in conflicts if c.severity == "warning"]

    if blocking:
        st.error("하드체크 위반으로 저장할 수 없습니다:")
        for c in blocking:
            st.write(f"- [{c.check_type}] {c.entity_id}: {c.reason}")
    for c in warnings:
        st.warning(f"[{c.check_type}] {c.entity_id}: {c.reason}")

    st.subheader("관련 기록")
    related_docs = st.session_state.detail_related_docs
    if not related_docs:
        st.write("관련 기록이 없습니다.")
    else:
        for doc in related_docs:
            doc_key = f"detail_flag_{entity_id}_{selected_field}_{doc.entity_id}"
            col1, col2 = st.columns([1, 8])
            checked = col1.checkbox("플래그", key=f"{doc_key}_check")
            col2.write(f"{doc.relevance_rank}. **{doc.entity_id}** ({doc.source}: {doc.relation})")
            col2.caption(doc.text)
            if checked:
                reason = col2.text_input("사유 (선택)", key=f"{doc_key}_reason")
                st.session_state.detail_flag_selection[doc.entity_id] = reason
            else:
                st.session_state.detail_flag_selection.pop(doc.entity_id, None)

    if st.button("저장", key="detail_save", disabled=bool(blocking)):
        if not field_update.is_structured_field(category, selected_field):
            storage.save_entity(category, entity_id, {selected_field: st.session_state.detail_new_value})
        if any(c.check_type == "lifespan" for c in warnings):
            storage.save_entity("character", entity_id, {"lifespan_check_ack": True})

        for flagged_entity_id, reason in st.session_state.detail_flag_selection.items():
            flags.add_flag(
                flagged_entity_id, f"{entity_id}의 {selected_field} 수정 중 발견", reason or None
            )

        cleared = flags.clear_flags_for_entity(entity_id)
        message = "저장 완료."
        if cleared:
            message += f" 이 엔티티에 걸려있던 플래그 {cleared}건이 자동 해제되었습니다."
        st.success(message)

        st.session_state.detail_searched = False
        st.session_state.detail_conflicts = []
        st.session_state.detail_related_docs = []
        st.session_state.detail_flag_selection = {}
        st.rerun()


# ---------------------------------------------------------------------------
# Top-level layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Lore Builder", layout="wide")
    _init_session_state()

    st.sidebar.title("Lore Builder")
    render_sidebar_search()
    st.sidebar.divider()

    mode = st.sidebar.radio("모드", ["채팅", "딕셔너리", "시각화"], key="mode")
    if mode == "시각화":
        st.sidebar.caption("곧 추가 예정")

    with st.sidebar.expander("🚩 플래그 확인"):
        deduped = flags.list_flags_deduped()
        if not deduped:
            st.write("플래그된 항목이 없습니다.")
        for flag in deduped:
            label = flag.reason or "(사유 없음)"
            if st.button(f"{flag.entity_id} — {label}", key=f"flagnav_{flag.id}"):
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
        st.info("시각화 모드는 곧 추가될 예정입니다.")


if __name__ == "__main__":
    main()
