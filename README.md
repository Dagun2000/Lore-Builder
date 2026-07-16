**Language:** English | [한국어](README.ko.md)

# Lore Builder

*A consistency-checking pipeline for fantasy worldbuilding.*

Lore Builder takes a single line describing something that just happened in your world — *"In 2100, [Jang] got into a brawl at [the Inn]"* — and checks it against everything already recorded before it's allowed into the database. Hard contradictions (a dead character reappearing, a destroyed item resurfacing) are rejected outright. Anything murkier — a possible world-rule violation, a clash with a character's established traits, even a numeric claim that doesn't add up against a stated correlation ("more circles = stronger" but this mage only has the minimum) — is surfaced for a human to decide, never applied automatically.

It keeps two kinds of knowledge in sync: structured facts in SQLite (entities and the timeline of events, ongoing relationships, and reversible statuses connecting them) and free-form descriptions in a Chroma vector store, so related lore can be found by meaning rather than by exact ID.

Every point-in-time occurrence and every ongoing relationship or personal status is a single `timeline` record, referenced from each entity it involves. An entity never stores a "current state" snapshot (no `current_owner`, no `current_status`); anything like that is computed on read from the timeline instead, since the world has no fixed notion of "now" — years can be entered in any order, spanning any range.

## Features

- **Natural-language event ingestion** — describe what happened in plain text with `[bracketed]` entity tags; the pipeline resolves each tag to an existing entity or walks you through creating a new one (category confirmation, required fields, optional attributes all auto-filled from context where possible).
- **Event-centric timeline** — a single `timeline` category holds both point-in-time occurrences (a fight, a founding) and ongoing duration records (a relationship, a reversible personal status like *imprisoned* or *cursed*), each one just an entity/predicate/target/start/end tuple. A cohesive multi-fact sentence can produce several linked records in one pass; a genuinely unrelated multi-event input is flagged for the user to split up instead of guessing.
- **Deterministic hard checks** — lifecycle consistency (no reappearing after your death year, no reappearing after being destroyed/disbanded) and race-lifespan cross-checks, with zero LLM involvement.
- **LLM-assisted inference** — turns a sentence into structured timeline records and, where the input introduces a new entity, splits out lifecycle attributes (birth/founding/creation year), persistent traits (notes), and any leftover time-bound event, anchored strictly to entities that already exist or are being created right here.
- **RAG cross-checks, on every new claim** — flags likely world-rule violations and contradictions with an entity's established notes or stored fields, reasoning about explicit constraints (access restrictions, ability requirements), danger/tone mismatches (a "life-threatening" location and a carefree picnic), and correlation self-contradictions (a rule says "more = stronger," but a character's own count is deep in the low end despite claiming to be exceptional — while still honoring any exception clause the rule itself states, like "usually hard to progress past 2"). This runs for dated events *and* for a bare new-entity introduction with no event at all — a brand-new entity's claims are checked before being trusted, exactly like an event would be. Every judgment carries a human-readable reason and a confirm/override prompt; nothing is ever silently rejected or silently accepted.
- **Review-before-write** — a change is assembled into one diff (the primary record plus whichever other entities get a pointer update alongside it) and shown as a single approve/reject decision, not one prompt per touched row.
- **Web GUI (Streamlit)** — a chat mode for the same natural-language pipeline, and a dictionary mode to browse every entity by category, open a detail view (current fields, a field editor with hard-check review and a 1-hop relevance search over connected entities), edit a timeline record directly, or delete an entity/event with its pointers cleaned up automatically.
- **CLI alternative** — `src/main.py` for the same chat pipeline, and `src/detail_panel.py` for existing-entity field edits, if you'd rather not run the GUI.
- **Flag for later** — mark a record surfaced during a review as needing a closer look, without blocking or auto-fixing anything; the flag clears automatically once that entity itself gets fixed.

## Requirements

- Python 3.12+
- An OpenAI API key (only needed for the inference/cross-check layers — schema, storage, and the deterministic hard checks run with no key at all)

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Web GUI

```bash
streamlit run app.py
```

The sidebar has a live search box (works no matter which mode is active) and two working modes: **채팅 / Chat** — the same natural-language event pipeline as the CLI, rendered as widgets instead of prompts — and **딕셔너리 / Dictionary** — browse every category, open any entity for its detail view (current fields, a field editor, related-context search with per-item flag checkboxes, delete), or open a timeline record to edit it in place or remove it. A third tab (visualization) is a placeholder for now.

### CLI: record a new event

```bash
python src/main.py
```

```
Lore Builder CLI. 종료하려면 '종료'를 입력하세요.

입력> 2100년, [미라]가 [은빛도시]에서 산책을 했다.
CREATE timeline: event_2100년_미라가_은빛도시에서_산책했다
  근거: 새 사건 기록: 2100년 미라가 은빛도시에서 산책했다.
  필드: {'year': 2100, 'location': 'loc_silver_city', 'notes': '2100년 미라가 은빛도시에서 산책했다.'}
  함께 갱신되는 엔티티: char_mira, loc_silver_city
저장하시겠습니까? [저장/취소]: 저장
저장 완료: event_2100년_미라가_은빛도시에서_산책했다, char_mira(갱신), loc_silver_city(갱신)
```

A new entity tag instead pauses for a category confirmation, then any required fields, before reaching the same review step. Type `종료` to exit.

### CLI: edit an existing entity

```bash
python src/detail_panel.py
```

Look up an entity by name or ID, pick a field, and enter a new value — every event this entity is pointed at is listed before anything saves (the GUI's field editor instead narrows this down to a relevance-judged 1-hop search). Type `목록` at any prompt to see everything flagged for later review.

### Seed the sample world

```bash
python scripts/seed_db.py
```

## Project Structure

```
schema_registry.yaml       # entity categories, fields, and types — the single source of truth
status_effects.yaml        # the fixed set of reversible statuses (imprisoned, cursed, lost, ...)
db/                         # seed data (Markdown with YAML frontmatter) + generated SQLite/Chroma stores
scripts/seed_db.py          # loads db/*.md into SQLite + Chroma
app.py                      # Streamlit GUI entry point

src/
  config.py                  # model-tier -> concrete model name mapping
  schema.py                   # schema_registry.yaml loader + lookup helpers
  storage.py                   # SQLite + Chroma storage layer
  hard_check.py                 # deterministic lifecycle/lifespan checks
  parser.py                      # regex-based input parsing
  mapping.py                      # entity tag resolution + new-entity creation
  inference.py                     # LLM event/attribute inference
  rag_check.py                      # LLM-assisted world-rule / notes / status cross-checks
  archivist.py                       # turns inference + checks into a reviewable diff
  approval.py                         # legacy blocking review loop (still used by main.run_pipeline)
  pipeline_session.py                  # generator-based pipeline state machine (pause/resume, shared by the GUI and the CLI's chat loop)
  deletion.py                           # entity/event deletion with pointer cleanup
  main.py                                # CLI entry point (chat loop + event pipeline)
  field_update.py                         # existing-entity field update logic (hard-check re-run + 1-hop relevance search)
  detail_panel.py                          # CLI entity-editing entry point
  flags.py                                  # "review later" bookkeeping

tests/                         # pytest suite (many tests call the OpenAI API)
```

## Testing

```bash
pytest
```

Tests covering the LLM-backed layers require `OPENAI_API_KEY` and make real API calls — run a targeted file or test rather than the whole suite by default. The deterministic layers (schema, storage, hard checks, diff assembly, flags) don't need a key.
