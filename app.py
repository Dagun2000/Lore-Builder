"""Lore Builder GUI — Phase 9 (Streamlit) + Phase 9 통합 패치.

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

from src import deletion, field_update, flags, hard_check, pipeline_session, schema, storage

_NAME_BEARING_CATEGORIES = ("character", "location", "faction", "artifact", "race")


# ---------------------------------------------------------------------------
# Session-state setup
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "session": None,
        "selected_entity": None,
        "_last_mode": None,
        "detail_field_name": None,
        "detail_previous_value": None,
        "detail_searched": False,
        "detail_conflicts": [],
        "detail_related_docs": [],
        "detail_new_value": None,
        "detail_flag_selection": {},
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
    st.session_state.detail_related_docs = []
    st.session_state.detail_flag_selection = {}
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
            options.append((f'{e["id"]} ({e.get("year", "?")}년)', e["id"]))
        return options

    entities = storage.list_entities(ref_category)
    if ref_category == "timeline":
        return [(f'{e["id"]} ({e.get("year", "?")}년)', e["id"]) for e in entities]
    return [(f'{e.get("name") or e["id"]} ({e["id"]})', e["id"]) for e in entities]


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
        choice = st.selectbox(display_name, ["(비어있음)"] + labels, index=index, key=key)
        return None if choice == "(비어있음)" else ids[labels.index(choice)]

    if field_type == "enum":
        options = field_def.get("options") or []
        index = options.index(current_value) + 1 if current_value in options else 0
        choice = st.selectbox(display_name, ["(비어있음)"] + options, index=index, key=key)
        return None if choice == "(비어있음)" else choice

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
        return f"입력 오류: {result['message']}"
    if status == "cancelled":
        return result.get("message", "취소되었습니다.")
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
    if status == "entity_only":
        return result.get("message", "엔티티가 저장되었습니다. 별도의 사건 기록은 없습니다.")
    if status == "no_new_info":
        return result.get("message", "새로 저장할 내용이 없습니다.")
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


def _render_entity_category_and_name(session, decision, key_prefix) -> None:
    """Phase 9 patch A: category confirmation is the headline here, not the
    name — a wrong category (person mistaken for an item) is the expensive
    mistake; the LLM rarely gets the name wrong."""
    payload = decision.payload
    categories = payload["categories"]
    st.write(
        f'"{payload["tag"]}"을(를) **{payload["inferred_category"]}**(으)로 분류했습니다. 맞습니까?'
    )
    category = st.selectbox(
        "카테고리",
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
        name = st.text_input("이름", value=payload["default_name"], key=f"{key_prefix}_name")

    col1, col2, col3 = st.columns(3)
    if col1.button("저장 후 계속", key=f"{key_prefix}_save"):
        _resume(session, {"category": category, "name": name, "action": "save"})
    if col2.button("편집", key=f"{key_prefix}_edit"):
        _resume(session, {"category": category, "name": name, "action": "edit"})
    if col3.button("취소", key=f"{key_prefix}_cancel"):
        _resume(session, {"category": category, "name": name, "action": "cancel"})


def _render_entity_terminal_status(session, decision, key_prefix) -> None:
    payload = decision.payload
    field_name = payload.get("field_name", "death_year")
    st.write(
        f"[{payload['tag']}]가 이 사건({payload['year']}년)으로 사망(또는 활동 종료)한 것으로 "
        f"추정됩니다. {field_name}={payload['year']}로 저장할까요?"
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
            f"새로운 {field_name} 값", value=payload["year"], step=1, key=f"{key_prefix}_year"
        )
        if st.button(f"{field_name}로 저장", key=f"{key_prefix}_edit_confirm"):
            _resume(session, {"수정": {field_name: int(new_year)}})


def _render_entity_required_field(session, decision, key_prefix) -> None:
    """Full field form (required forced server-side, optional included) —
    every field goes through the same type -> widget mapping as the
    entity-detail editor (Phase 9 patch B)."""
    payload = decision.payload
    st.write(f"[{payload['category']}] 필드를 입력하세요 (필수 항목은 *):")
    values = {}
    for f in payload["fields"]:
        label = f"{f['name']} *" if f["required"] else f["name"]
        values[f["name"]] = _render_value_field(
            f, None, key=f"{key_prefix}_{f['name']}", label=label
        )
    if st.button("저장", key=f"{key_prefix}_submit"):
        _resume(session, values)


def _render_hard_check_warning(session, decision, key_prefix) -> None:
    """Phase 10 patch 7 (E): "수정" used to sit between these two buttons
    but never actually offered any editing — it just fell through to the
    same rejection "그래도 저장"'s absence already causes. Two honest
    options instead of three, one of which lied about what it did."""
    payload = decision.payload
    st.warning(f"[{payload['entity_id']}] {payload['reason']}")
    col1, col2 = st.columns(2)
    if col1.button("그래도 저장", key=f"{key_prefix}_accept"):
        _resume(session, "그래도 저장")
    if col2.button("취소", key=f"{key_prefix}_cancel"):
        _resume(session, "취소")


def _render_rag_judgment(session, decision, key_prefix) -> None:
    payload = decision.payload
    st.warning(f"[{payload['judgment_type']}] {payload['reason']}")
    col1, col2 = st.columns(2)
    if col1.button("그래도 저장", key=f"{key_prefix}_accept"):
        _resume(session, "그래도 저장")
    if col2.button("취소", key=f"{key_prefix}_cancel"):
        _resume(session, "취소")


def _render_diff_review(session, decision, key_prefix) -> None:
    """One bundled decision for the whole diff (Phase 10 patch) — the
    primary record plus whichever other entities get an event_ids/cache
    update alongside it, shown as information only. No per-item toggling,
    no edit here: 저장 applies everything, 취소 applies nothing."""
    payload = decision.payload
    st.write(f"**{payload['action'].upper()} {payload['category']}**: {payload['entity_id']}")
    st.caption(f"근거: {payload['reason']}")
    st.json(payload["fields"])
    if payload["affected_entities"]:
        st.caption("함께 갱신되는 엔티티: " + ", ".join(payload["affected_entities"]))
    col1, col2 = st.columns(2)
    if col1.button("저장", key=f"{key_prefix}_save"):
        _resume(session, True)
    if col2.button("취소", key=f"{key_prefix}_cancel"):
        _resume(session, False)


def _render_multi_event_warning(session, decision, key_prefix) -> None:
    """Nothing gets saved here either way (see pipeline_session's identical
    comment) — this is purely an acknowledgment, not a choice between two
    outcomes that both do the same non-thing."""
    payload = decision.payload
    st.warning(f"[확인 필요] {payload['reason']}")
    st.caption("저장된 내용이 없습니다. 입력을 나눠서 다시 시도해주세요.")
    if st.button("확인", key=f"{key_prefix}_ack"):
        _resume(session, None)


_DECISION_RENDERERS = {
    "entity_candidates": _render_entity_candidates,
    "entity_category_and_name": _render_entity_category_and_name,
    "entity_terminal_status": _render_entity_terminal_status,
    "entity_required_field": _render_entity_required_field,
    "multi_event_warning": _render_multi_event_warning,
    "hard_check_warning": _render_hard_check_warning,
    "rag_judgment": _render_rag_judgment,
    "diff_review": _render_diff_review,
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


def _dictionary_label(category: str, entity: dict) -> str:
    """Phase 10 patch 8: the id is a content-derived slug (often just the
    name/notes with spaces swapped for underscores) — never useful for a
    person to look at, so it's dropped everywhere except as a fallback for
    the one category with no `name` field at all (timeline)."""
    if category == "timeline":
        if entity.get("year") is not None:
            return f"[{entity['year']}년] {entity.get('notes') or entity['id']}"
        start = entity.get("start_year")
        span = f"{start}~{entity.get('end_year') or '현재'}" if start is not None else "?"
        predicate = entity.get("predicate") or ""
        return f"[{span}] {predicate} — {entity.get('notes') or entity['id']}"
    return entity.get("name") or entity["id"]


_STATUS_EFFECTS_PSEUDO_CATEGORY = "상태 효과"


def _status_effect_usage_count(effect_id: str) -> int:
    """How many timeline duration records currently use effect_id as their
    (target-less, personal-status) predicate — shown before deletion so
    removing a status doesn't silently orphan existing records from every
    check/dropdown that reads status_effects.yaml going forward."""
    return sum(
        1
        for e in storage.list_entities("timeline")
        if e.get("predicate") == effect_id and e.get("target") is None
    )


def _render_status_effects_panel() -> None:
    """A GUI-editable status_effects.yaml (Phase 10 patch 14) — the set of
    reversible statuses (imprisoned, cursed, ...) a world can have is a
    setting-specific choice, not something the code should hardcode; a
    sci-fi setting might add "cryosleep" the same way a fantasy one added
    "imprisoned". Lives as a pseudo-category in the dictionary rather than a
    real schema_registry.yaml category, since status effects aren't
    entities — they never get their own id/detail screen, just this list."""
    st.write(
        "세계관에서 쓸 수 있는, 되돌릴 수 있는 상태(수감, 저주 등)의 목록입니다. "
        "새 사건 입력이나 필드 수정 화면에서 바로 선택지로 나타납니다."
    )

    effects = schema.load_status_effects()
    pending = st.session_state.status_effect_confirm_delete

    for effect in effects:
        effect_id = effect["id"]
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{effect_id}** ({effect['label']})")

        if pending != effect_id:
            if col2.button("삭제", key=f"status_effect_delete_{effect_id}"):
                st.session_state.status_effect_confirm_delete = effect_id
                st.rerun()
            continue

        usage = _status_effect_usage_count(effect_id)
        if usage:
            st.warning(
                f"현재 {usage}건의 기록이 이 상태를 사용하고 있습니다. 삭제해도 그 기록 "
                "자체는 남지만, 앞으로 선택지에 나타나지 않고 상태 일관성 검증 대상에서도 "
                "빠지게 됩니다."
            )
        confirm_col1, confirm_col2 = st.columns(2)
        if confirm_col1.button("그대로 삭제", key=f"status_effect_delete_confirm_{effect_id}"):
            schema.remove_status_effect(effect_id)
            st.session_state.status_effect_confirm_delete = None
            st.success(f"'{effect_id}' 상태를 삭제했습니다.")
            st.rerun()
        if confirm_col2.button("취소", key=f"status_effect_delete_cancel_{effect_id}"):
            st.session_state.status_effect_confirm_delete = None
            st.rerun()

    st.divider()
    st.subheader("새 상태 추가")
    with st.form("status_effect_add_form", clear_on_submit=True):
        new_id = st.text_input("id (코드에서 predicate로 쓰일 값, 영문 권장)", key="status_effect_new_id")
        new_label = st.text_input("표시 이름", key="status_effect_new_label")
        submitted = st.form_submit_button("추가", key="status_effect_add")
    if submitted:
        try:
            schema.add_status_effect(new_id, new_label)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(f"'{new_id}' ({new_label}) 상태를 추가했습니다.")
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
    st.header("딕셔너리")
    categories = schema.list_categories() + [_STATUS_EFFECTS_PSEUDO_CATEGORY]
    remembered = st.session_state.dict_category_persist
    default_index = categories.index(remembered) if remembered in categories else 0
    category = st.selectbox("카테고리", categories, index=default_index, key="dict_category")
    st.session_state.dict_category_persist = category

    if category == _STATUS_EFFECTS_PSEUDO_CATEGORY:
        _render_status_effects_panel()
        return

    entities = storage.list_entities(category)
    if not entities:
        st.write("이 카테고리에 등록된 엔티티가 없습니다.")
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


def _render_relevant_context_section(entity_id: str, field_name: str, new_value) -> list:
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
    """
    st.subheader("관련 기록")
    matches = field_update.find_relevant_context(entity_id, field_name, new_value)
    if not matches:
        st.write("관련성이 있어 보이는 기록이 없습니다.")
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
                "플래그", key=f"flag_{entity_id}_{field_name}_{match.entity_id}"
            )
        with col2:
            if st.button(f"{match.entity_id} 상세 보기", key=f"relctx_{entity_id}_{field_name}_{match.entity_id}"):
                _navigate_to_entity(match.entity_id)
                st.rerun()

    if remaining:
        if st.button(f"더 보기 ({remaining}건 더 있음)", key=f"relctx_more_{entity_id}_{field_name}"):
            st.session_state.detail_relevant_show_all = True
            st.rerun()

    return matches


def _render_field_editor_section(category: str, entity_id: str, entity: dict) -> None:
    field_defs = [f for f in schema.get_fields(category) if f["name"] != "event_ids"]
    field_names = [f["name"] for f in field_defs]

    st.subheader("필드 수정")
    selected_field = st.selectbox("필드 선택", field_names, key="detail_field_select")

    if st.session_state.detail_field_name != selected_field:
        _rollback_pending_structured_write()
        st.session_state.detail_field_name = selected_field
        st.session_state.detail_previous_value = entity.get(selected_field)
        st.session_state.detail_searched = False
        st.session_state.detail_conflicts = []
        st.session_state.detail_flag_selection = {}
        st.session_state.detail_relevant_show_all = False

    field_def = next(f for f in field_defs if f["name"] == selected_field)
    new_value = _render_value_field(
        field_def, entity.get(selected_field), key=f"detail_value_{entity_id}_{selected_field}"
    )

    if st.button("필드 값 검토", key="detail_search"):
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
    if not conflicts and field_def.get("role") in ("lifecycle_start", "lifecycle_end"):
        # Phase 10 patch 7 follow-up: a lifecycle field's only meaningful
        # "related event" question is "does this value conflict with a
        # recorded year" — hard_check already answers that above when it
        # fires. When it doesn't, say so explicitly instead of leaving a
        # blank gap where a warning would otherwise have been.
        st.success("타임라인 충돌이 감지되지 않았습니다.")

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

    if st.button("저장", key="detail_save", disabled=bool(blocking)):
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

        message = "저장 완료."
        if flagged_count:
            message += f" {flagged_count}건 플래그 등록."
        st.success(message)

        st.session_state.detail_searched = False
        st.session_state.detail_conflicts = []
        st.session_state.detail_flag_selection = {}
        st.session_state.detail_relevant_show_all = False
        st.rerun()


def _render_delete_entity_section(category: str, entity_id: str) -> None:
    st.subheader("엔티티 삭제")
    if not st.session_state.detail_confirm_delete:
        if st.button("이 엔티티 삭제", key="detail_delete_start"):
            st.session_state.detail_confirm_delete = True
            st.rerun()
        return

    events = deletion.request_entity_deletion(entity_id)
    if events:
        st.write(f"이 엔티티가 관여한 이벤트 {len(events)}건도 함께 정리됩니다 (다른 엔티티가 "
                 f"관여하지 않은 이벤트는 삭제, 관여했다면 이 엔티티의 포인터만 제거):")
        for record in events:
            st.write(f"- {record['id']}: {record.get('notes') or ''}")

    col1, col2 = st.columns(2)
    if col1.button("그대로 삭제 진행", key="detail_delete_confirm"):
        result = deletion.delete_entity(entity_id, category)
        st.session_state.detail_confirm_delete = False
        _navigate_to_entity(None)
        message = f"{result.deleted_id} 삭제 완료."
        if result.deleted_events:
            message += f" 함께 삭제된 이벤트: {', '.join(result.deleted_events)}."
        if result.affected_entities:
            message += f" 포인터가 갱신된 엔티티: {', '.join(result.affected_entities)}."
        st.success(message)
        st.rerun()
    if col2.button("취소 (유지)", key="detail_delete_cancel"):
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

    st.subheader("이벤트 수정")
    if is_point:
        new_year = st.number_input("연도", value=entity.get("year"), step=1, key=f"tl_year_{entity_id}")
        loc_options = [(f'{e.get("name") or e["id"]}', e["id"]) for e in storage.list_entities("location")]
        loc_ids = [eid for _label, eid in loc_options]
        loc_labels = ["(없음)"] + [label for label, _eid in loc_options]
        loc_index = loc_ids.index(entity.get("location")) + 1 if entity.get("location") in loc_ids else 0
        loc_choice = st.selectbox("장소", loc_labels, index=loc_index, key=f"tl_loc_{entity_id}")
        new_location = None if loc_choice == "(없음)" else loc_ids[loc_labels.index(loc_choice) - 1]

        selected_participants = st.multiselect(
            "참가자", candidates, default=[p for p in previous_participants if p in candidates],
            format_func=lambda eid: labels_by_id.get(eid, eid), key=f"tl_participants_{entity_id}",
        )
    else:
        new_start = st.number_input("시작 연도", value=entity.get("start_year"), step=1, key=f"tl_start_{entity_id}")
        new_end = st.number_input("종료 연도 (비워두면 현재도 진행 중)", value=entity.get("end_year"), step=1, key=f"tl_end_{entity_id}")
        new_predicate = st.text_input("predicate (상태/관계 이름)", value=entity.get("predicate") or "", key=f"tl_pred_{entity_id}")
        entity_index = candidates.index(entity.get("entity")) if entity.get("entity") in candidates else 0
        new_entity = st.selectbox(
            "주체 (entity)", candidates, index=entity_index,
            format_func=lambda eid: labels_by_id.get(eid, eid), key=f"tl_entity_{entity_id}",
        ) if candidates else None
        target_labels = ["(없음)"] + [labels_by_id[eid] for eid in candidates]
        target_index = candidates.index(entity.get("target")) + 1 if entity.get("target") in candidates else 0
        target_choice = st.selectbox("대상 (target, 관계형일 때만)", target_labels, index=target_index, key=f"tl_target_{entity_id}")
        new_target = None if target_choice == "(없음)" else candidates[target_labels.index(target_choice) - 1]
        selected_participants = [p for p in (new_entity, new_target) if p]

    new_notes = st.text_area("비고", value=entity.get("notes") or "", key=f"tl_notes_{entity_id}")

    review_key = f"tl_reviewed_{entity_id}"
    pending_key = f"tl_pending_{entity_id}"

    if st.button("변경사항 검토", key=f"tl_review_{entity_id}"):
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
        st.error("하드체크 위반으로 저장할 수 없습니다:")
        for c in blocking:
            st.write(f"- [{c.check_type}] {c.entity_id}: {c.reason}")
    for c in warnings:
        st.warning(f"[{c.check_type}] {c.entity_id}: {c.reason}")
    if not conflicts:
        st.success("하드체크 충돌이 감지되지 않았습니다.")

    if st.button("저장", key=f"tl_save_{entity_id}", disabled=bool(blocking)):
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
        st.success("이벤트가 갱신되었습니다.")
        st.rerun()

    st.subheader("이벤트 삭제")
    if st.button("이 이벤트 삭제", key=f"tl_delete_{entity_id}"):
        result = deletion.delete_event(entity_id)
        message = f"{result.deleted_id} 삭제 완료."
        if result.affected_entities:
            message += " 포인터가 제거된 엔티티: " + ", ".join(result.affected_entities)
        st.success(message)
        _navigate_to_entity(None)
        st.rerun()


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

    if category == "timeline":
        # Phase 10 patch 8, section 4: an event has no fields that can be
        # patched one at a time without invalidating its own content-derived
        # id, so it gets a dedicated form instead of the generic field
        # editor — see _render_timeline_detail.
        _render_timeline_detail(entity_id, entity)
        return

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
            _render_relevant_context_section(
                entity_id, st.session_state.detail_field_name, st.session_state.detail_new_value
            )
        _render_save_section(category, entity_id, st.session_state.detail_field_name)
    _render_delete_entity_section(category, entity_id)


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

    # Phase 9 patch E: clicking a different mode tab must win over "an
    # entity detail screen happens to be open" — previously the
    # selected_entity check below ran unconditionally every rerun and never
    # noticed the mode had changed, so switching tabs while viewing an
    # entity did nothing until you clicked "← 목록으로" first.
    if st.session_state._last_mode is not None and st.session_state._last_mode != mode:
        _navigate_to_entity(None)
    st.session_state._last_mode = mode

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
