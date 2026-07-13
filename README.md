세계관이 스스로를 검열하는 파이프라인
Lore Reviewer는 "대륙력 2100년, [쟝]이 [주점]에서 얻어맞았다" 같은 한 줄짜리 사건 기록을 입력받아, 그 사건이 지금까지 쌓인 설정과 모순되지 않는지 확인한 뒤에만 SQLite + Chroma 저장소에 반영하는 CLI 파이프라인이다. 결정론적 검증(하드체크)과 LLM 판단(RAG 교차검증)을 분리해서, 확실히 틀린 것은 즉시 반려하고 애매한 것만 사람에게 묻는다.

Phase 0 스키마/저장소
→
Phase 1 하드체크
→
Phase 2 파싱/매핑
→
Phase 3 추론/RAG
→
Phase 4 Archivist
→
Phase 5 승인/저장
Phase 0
스키마 레지스트리 & 저장소
schema_registry.yaml · status_effects.yaml · src/schema.py · src/storage.py · db/*.md

8개 카테고리, 하나의 YAML이 전부를 정의한다
모든 엔티티 종류(character, race, location, faction, artifact, timeline, relationship, system)는 schema_registry.yaml 하나에 id_prefix와 필드 목록으로 정의된다. 필드에는 type(integer/text/boolean/enum/reference/list) 외에 선택적으로 role이 붙는데, 이 role이 이후 모든 하드체크·diff 로직이 "어떤 필드를 봐야 하는지" 하드코딩 없이 찾아내는 열쇠다.

role	쓰이는 곳	예시 필드
lifecycle_start / lifecycle_end	Phase 1 시간축 검증	birth_year / death_year, founded_year / destroyed_year
timeline_point	사건의 기준 연도	timeline.year
ownership	아이템 소유권(예약, 미사용)	artifact.current_owner
이중 저장소: SQLite(구조) + Chroma(서술)
같은 엔티티가 두 곳에 동시에 저장된다. SQLite는 카테고리별 테이블(스키마에서 동적으로 DDL 생성)에 정형 필드를 담고, Chroma는 사람이 쓴 서술 텍스트(body)를 임베딩해서 "이 사건과 비슷한 기존 기록"을 찾는 데 쓰인다. 하나는 규칙이 읽고, 하나는 LLM이 읽는다.

storage.py 핵심 함수
save_entity(category, entity_id, fields) → None
upsert. 기존 레코드를 읽어 fields를 머지한 뒤 덮어써서, 부분 업데이트가 나머지 필드를 지우지 않는다.

get_event_years(entity_id) → list[int]
이 엔티티와 관련된 모든 timeline 연도. timeline.location 직접 참조 + relationship을 경유한 간접 연결(subject/object가 이 엔티티고 상대가 event_인 경우) 둘 다 스캔한다.

get_status_effects(entity_id) → list[str]
현재 활성 상태 id 목록. Phase 4 이후로는 active_status_effects 필드를 직접 읽기만 한다 — 원래는 timeline/relationship을 스캔해서 "해제 안 된 상태"를 역산했는데, relationship이 append-only 이력이라 해제 여부를 절대 구분할 수 없다는 게 드러나서 스냅샷 필드 읽기로 리팩터링했다.

Phase 1
하드체크 — 결정론적, LLM 없음
src/hard_check.py

Conflict
Conflict(check_type: "terminal"|"lifespan", severity: "blocking"|"warning", entity_id, reason)
severity="blocking"은 확인 없이 즉시 반려된다. "warning"만 Phase 5의 승인 루프에서 사람에게 [그래도 저장/수정/취소]를 묻는다 — 확실히 틀린 것과 애매한 것을 이 한 필드가 가른다.

두 가지 검증
check_terminal_violation(category, entity_id, extra_years=None)
lifecycle_start ≤ min(사건연도) 그리고 lifecycle_end ≥ max(사건연도). 어긴 즉시 blocking. character든 artifact든 location이든 role만 보고 동작해서, 죽은 인물의 재등장과 파괴된 유물의 재등장을 같은 코드가 잡는다.

check_lifespan_violation(character_id, extra_years=None)
종족 lifespan 대비 (death_year 또는 사건연도 최댓값) − (birth_year 또는 사건연도 최솟값). 초과하면 warning. lifespan_check_ack=True면 무조건 스킵 — 한 번 확인받은 인물은 다시 안 묻는다.

Phase 5에서 발견한 버그
원래 시그니처엔 extra_years가 없었다. 문제는 파이프라인 순서: 하드체크(Step 5)가 diff 생성(Step 6)보다 먼저 도는데, get_event_years()는 이미 저장된 연도만 본다. 그래서 "이미 죽은 인물이 재등장" 같은, 지금 막 입력된 사건의 연도가 있어야 잡히는 케이스를 못 잡았다. 저장 전 후보 연도를 얹어 볼 수 있게 옵션 파라미터를 추가해서 해결 — 기본값 None이라 기존 호출부/테스트는 그대로다.
Phase 2
입력 파싱 & 엔티티 매핑
src/parser.py · src/mapping.py · 모델: gpt-5.4-mini (config.get_model("simple"))

parser.parse_input(text) → ParsedInput
정규식만 쓴다 — LLM 없음. [대괄호]로 감싼 조각은 전부 태그로, 숫자+년 패턴은 연도로 뽑는다. 연도가 없으면 ValueError를 던져서, 파싱 단계와 LLM 추론 단계의 실패를 헷갈리지 않게 분리해둔다.

태그 하나가 entity_id로 확정되기까지 — resolve_entity()
resolve_entity(tag, context_sentence, year) → entity_id

1. infer_category(tag, context)         — LLM, 8개 카테고리 중 하나
2. find_existing_matches(tag, category) — 규칙기반, id suffix 정확일치 → notes/appearance 부분일치
     0건 → 신규 생성 흐름
     1건 → 자동 확정, 확인 없음
     N건 → 번호로 후보 선택 (CLI)
3. (신규일 때) character면 LLM으로 "이 문장이 죽음/영구 종료를 암시하는가" 판단
     → 예/아니오/수정 3지선다로 death_year 제안
   그 외 필수 필드는 값이 들어올 때까지 Enter로 스킵 불가
동음이의어 버그 — "미라"
"미라"는 사람 이름이면서 동시에 "미라(mummy)"라는 일반명사다. 실제로 infer_category가 가끔 이를 race로 오분류해서, 기존 캐릭터를 매칭하는 대신 새 종족 엔티티를 만들어버렸다. "롱리브드", "데드맨"처럼 서술적으로 들리는 이름도 같은 방식으로 오분류됐다. 프롬프트에 "태그가 문장에서 행동의 주체(~가/이 ~했다)로 나오면 character를 우선하라"는 명시적 소거 규칙을 추가해서 완화했다 — 실사용자도 겪을 수 있는 문제라 테스트 우회가 아니라 프롬프트 자체를 고쳤다.
필수 필드는 절대 스킵될 수 없다
_collect_fields()는 스키마상 required: true인 필드를 값이 들어올 때까지 강제로 재질문한다 — Enter로 넘어갈 수 있는 건 optional 필드뿐이다. 처음엔 required도 Enter로 스킵 가능했는데, 그러면 Phase 1 하드체크가 애초에 검증할 값 자체가 없어서 조용히 다 통과하는 구멍이 생긴다는 지적을 받고 고쳤다.
Phase 3
관계/사건 추론 & RAG 교차검증
src/inference.py · src/rag_check.py · 모델: gpt-5.6-terra (config.get_model("reasoning"))

infer_relationship_and_event(resolved_entities, raw_text, year) → InferredEvent
이미 확정된 entity_id 목록을 프롬프트에 앵커로 박아 넣고, LLM에게 "이 사이에 무슨 일이 있었는가"만 추론시킨다 — 새 엔티티를 지어내는 건 명시적으로 금지. JSON 강제 출력, 파싱 실패 시 1회 재시도.

InferredEvent(
  event_summary: str,
  relationships: [{subject, predicate, object}, ...],
  status_effect: {entity, effect, action: "set"|"clear"} | None
)
status_effect.effect는 status_effects.yaml의 5개 id (sealed / imprisoned / missing / cursed / incapacitated)로 강하게 제약된다.

존재하지 않는 상태를 지어낸 버그
제약을 걸기 전엔 "쟝이 얻어맞았다"에 LLM이 effect: "injured"를 지어냈다 — 우리 상태 목록에 없는 값이다. 프롬프트에 실제 5개 id를 명시하고 "단순한 몸싸움 정도로는 채우지 마라"는 기준을 추가해서 해결했다.
run_rag_checks(entities, raw_text) → list[Judgment]
Judgment(type, reason, confidence=None, entity_id=None, status_effect_id=None)
check_rule_violation(raw_text, hard_rule_docs)
system.hard_rule=true 규칙 전체를 SQLite에서 직접 가져와 대조한다. 임베딩 유사도 검색 결과는 절대 섞지 않는다 — 세계관 규칙은 작고 확정적인 목록이라, 무관한 검색 결과가 프롬프트를 희석시켜 진짜 위반을 놓치게 만드는 걸 실제로 확인했다.

check_notes_conflict(entities, raw_text)
엔티티 notes + (character면) 소속 race의 notes까지 모아서 모순을 판단. "고기를 먹지 않는다"는 종족 설정과 "고기를 먹었다"는 사건을 여기서 잡는다.

check_status_consistency(entity_id, raw_text)
storage.get_status_effects()로 현재 활성 상태를 읽고, 이번 사건이 그 상태와 양립하는지(ok) / 해제하는지(clears) / 상충하는지(conflict) 판단.

RAG 노이즈가 규칙위반 탐지를 방해한 버그
run_rag_checks가 원래 check_rule_violation에 일반 유사도검색 결과와 확정 규칙을 섞어서 넘겼다. "손끝에서 불꽃을 만들어냈다"처럼 짧은 문장은 세션에 쌓인 무관한 문서(예: 다른 캐릭터의 탈출 사건)를 끌어와서, 정작 "마나 스톤 없이는 마법 시전 불가" 규칙 텍스트를 희석시켜 위반을 놓치는 일이 재현됐다. 규칙위반 체크는 확정 규칙 텍스트만 받도록 분리해서 5/5 안정적으로 잡도록 고쳤다.
Phase 4
Archivist — diff 조립
src/archivist.py · LLM 호출 없음, 순수 조립

build_diff(parsed, resolved_entities, inferred_event, rag_judgments) → list[ChangeItem]
ChangeItem(action: "create"|"update", category, entity_id, fields, body, reason)
timeline 레코드는 항상 신규 생성. id는 generate_id()로 prefix + slugify(event_summary), 충돌 시 _2, _3… 접미사.
추론된 relationship마다 항상 신규 rel_ 레코드 생성. 절대 update 없음 — relationship은 append-only 이력 로그다.
status_effect가 set/clear되면 해당 엔티티(character/location/faction/artifact)의 active_status_effects 필드 하나만 update.
왜 relationship.until이 아니라 새 필드인가
"상태 해제"를 relationship의 until을 고쳐서 표현하는 방법도 검토했지만, 그러면 relationship이 "항상 create"라는 원칙과 정면충돌하고 이력 로그를 수정 가능한 상태로 바꿔버린다. 대신 character/location/faction/artifact에 active_status_effects (list) 필드를 새로 추가해서, "현재 상태"는 엔티티의 스냅샷으로 relationship은 이력으로 역할을 분리했다 — 두 요구사항이 충돌 없이 동시에 성립한다.
Phase 5
승인 루프 & 전체 파이프라인
src/approval.py · src/main.py

approval.py — 세 개의 승인 루프
review_hard_check_conflicts(conflicts) → bool
blocking 하나라도 있으면 이유 출력 후 즉시 False, 확인 없음. warning은 항목별 [그래도 저장/수정/취소] — "그래도 저장"이 lifespan 경고면 그 자리에서 lifespan_check_ack=True를 저장소에 바로 반영한다.

review_rag_judgments(judgments) → bool
clears_status는 안내만 하고 자동 통과. 그 외(conflict/notes_conflict/rule_violation)는 전부 동일하게 [그래도 저장/수정/취소]를 묻는다.

review_diff(diff) → list[ChangeItem]
diff 항목을 순서 그대로 하나씩 보여주고 y/n. 승인된 것만 모아 반환.

main.run_pipeline(user_input) — 8단계
1 parser.parse_input
2 각 태그 → mapping.resolve_entity            → resolved_entities
3 inference.infer_relationship_and_event      → inferred_event
4 rag_check.run_rag_checks                    → rag_judgments
5 각 엔티티 → hard_check.run_hard_checks(extra_years=[연도])
  → approval.review_hard_check_conflicts   실패 시 status="rejected" 중단
  → approval.review_rag_judgments          실패 시 status="rejected" 중단
6 archivist.build_diff                        → diff
7 approval.review_diff                        → approved
8 approved를 create부터, 그다음 update 순으로 SQLite+Chroma 반영
run_pipeline은 print만 하고 끝나지 않고 {status, resolved_entities, diff, applied, ...} 구조화된 dict를 반환한다 — 나중에 GUI가 만들어지면 표준출력을 긁어읽지 않고 이 반환값만으로 화면을 그릴 수 있게 하기 위함.

실행 사례
실제로 돌려본 세 가지 결과
아래 세 건은 시뮬레이션이 아니라 python src/main.py를 실제로 실행해 얻은 원본 출력이다.

"2100년, [쟝]이 [주점]에서 술을 마셨다."
저장됨
Step 1 — parse
year=2100, tags=["쟝","주점"]
Step 2 — resolve × 2
"쟝"은 char_jang notes와 부분일치 → 즉시 확정, 확인 없음.
"주점"은 loc_black_goat_inn notes("…허름한 주점…")와 부분일치 → 즉시 확정.
Step 3 — infer
event_summary 생성, relationship 1건: char_jang —술을 마셨다→ loc_black_goat_inn. status_effect 없음(단순 음주는 5개 상태 중 어디에도 안 걸림).
Step 4 — RAG
규칙위반·notes모순·상태충돌 전부 해당 없음 → judgments 없음.
Step 5 — 하드체크
char_jang lifespan 100년 vs 나이 정확히 100년 — age > lifespan이 아니라 통과. conflicts 없음, 팝업 없음.
Step 6/7 — diff 승인
[1/2] CREATE timeline: event_2100년_쟝이_주점에서_술을_마셨다_2
  필드: {'year': 2100, 'location': 'loc_black_goat_inn', ...}
  승인하시겠습니까? (y/n): y

[2/2] CREATE relationship: rel_char_jang_술을_마셨다_loc_black_goa
  근거: char_jang와(과) loc_black_goat_inn 사이의 '술을 마셨다' 관계 기록
  승인하시겠습니까? (y/n): y
Step 8 — 반영
SQLite 2건(timeline, relationship) + 각 body를 Chroma에 upsert. 저장 완료: event_..., rel_...
"2350년, [레오]가 마을 광장에 나타났다." (레오는 2300년에 사망 처리됨)
즉시 반려
Step 2 — resolve
"레오" → char_레오 즉시 확정 (death_year=2300 보유).
Step 5 — 하드체크
extra_years=[2350]이 합쳐지며 check_terminal_violation이 death_year(2300) < 관련 사건 연도(2350)를 감지 → severity=blocking.
즉시 중단 — 확인 팝업 없음
다음 항목이 하드체크를 위반해 저장이 거부되었습니다:
  - [terminal] char_레오: char_레오의 death_year(2300)이(가)
    관련 사건 연도(2350)보다 이릅니다.
하드체크 결과에 따라 저장이 중단되었습니다.
Step 3(추론)·4(RAG)는 이미 돌았지만 diff는 아예 만들어지지 않았다 — SQLite에 아무것도 쓰이지 않음.
"2100년, [핍]이 시장에 나타났다." (핍의 종족 수명 30년, 실제 나이 100년)
경고 → 그래도 저장
Step 5 — 하드체크 경고
[경고] char_demo_pip: char_demo_pip(종족: race_demo_short)의
생존 기간이 2000~2100년으로 총 100년입니다.
종족 수명 30년을 70년 초과했습니다.
그래도 저장하시겠습니까? [그래도 저장/수정/취소]: 그래도 저장
부수 효과
"그래도 저장" 선택 즉시 char_demo_pip.lifespan_check_ack = True가 저장소에 반영됨. 이후 같은 인물이 다시 수명을 초과해도 이 팝업은 다시 뜨지 않는다.
Step 6~8
경고는 차단이 아니므로 diff 생성·승인·반영은 정상 진행되어 최종 저장 완료.
현재 알려진 한계
승인 루프가 전부 블로킹 input() 기반이다. GUI로 옮기려면 "결정이 필요하면 멈추고, 응답이 오면 재개하는" 상태 머신으로 파이프라인을 재설계해야 할 가능성이 높다. run_pipeline이 구조화된 dict를 반환하도록 만들어 두긴 했지만, 그 자체가 상태 머신은 아니다.
check_rule_violation은 여전히 확률적이다. 프롬프트 보강으로 5/5까지 끌어올렸지만 근본적으로 결정론적이지 않다.
find_existing_matches는 정확/부분 문자열 일치만 한다. notes에 없는 별칭·구어체 표현은 매칭되지 않고 신규 엔티티로 새로 만들어진다 — 임베딩 유사도 매칭은 의도적으로 Phase 2 범위 밖으로 미뤄뒀다.
