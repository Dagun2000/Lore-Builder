**Language:** English | [한국어](README.ko.md)

# Lore Builder

*Keep a fictional world's lore consistent, one sentence at a time.*

Lore Builder is a consistency-checking assistant for building out a fictional world's history and cast of characters. Describe an event, a fact, or a relationship in plain language — tag the entities involved in `[brackets]` — and the pipeline checks it against everything already on record before it's saved:

```
[Jang] got into a brawl at [the Inn] in 2100.
```

Outright contradictions (a dead character reappearing, a destroyed item resurfacing, a location's own notes disagreeing with what's being described) are rejected automatically. Anything murkier — a possible world-rule violation, a clash with a character's established traits, a number that doesn't add up against a stated correlation — is surfaced with a plain-language reason and left for you to decide. Nothing is ever saved or rejected silently.

Entities and their history live side by side: structured facts in SQLite (who exists, what happened, when, and how it's all connected) and free-form descriptions in a vector store, so related lore can be found by meaning, not just by exact name.

## Why

A world's lore gets hard to hold in your head once there's enough of it — an exiled character wandering back into the capital, a sword forged after its wielder died, a rule about who can join a guild that a new member quietly breaks. Lore Builder doesn't write your story for you; it remembers everything you've already established and flags anything new that doesn't fit, before it quietly becomes a plot hole three chapters later.

## Features

- **Natural-language entry** — describe events and facts in plain text, tagging entities with `[brackets]`. An unrecognized tag walks you through creating a new entity: category, required fields, and as many optional details as can be inferred from context.
- **A timeline, not a wiki page** — every event, relationship, and reversible status (imprisoned, cursed, exiled, ...) is one entry in a shared timeline, linked to everyone it involves. Nothing stores a "current state" snapshot — status is always computed from the timeline itself, since the world has no fixed sense of "now": history can be added in any year, in any order.
- **Deterministic checks** — lifecycle consistency (no reappearing after a death or destruction date) and race/lifespan cross-checks, resolved with plain logic, no model call needed.
- **LLM-assisted understanding** — turns a sentence into structured facts (who did what, to whom, when), and, for a new entity, separates out lifecycle attributes, persistent traits, and any event still worth recording in its own right.
- **Contradiction checks on everything new** — flags likely rule violations and clashes with an entity's own established notes: explicit constraints, tone mismatches (a "life-threatening" ruin and a carefree picnic), even a number that doesn't hold up against a stated correlation ("more means stronger," but this one barely qualifies) — while still honoring any exception the rule itself allows for. This runs for a dated event and for a bare new-entity introduction alike, always with a reason and a chance to override.
- **One decision per change** — a change is bundled into a single diff (the core fact plus whatever else needs updating alongside it) and shown as one approve/reject choice, not one prompt per row touched.
- **Registries you control** — the set of reversible statuses and relationship types (imprisoned, exiled, allied-with, ...) is a plain editable list, not hardcoded; a genuinely new one Step 3 needs gets proposed to you for approval before it's added.
- **Web GUI** — chat for the same natural-language flow, plus a browsable dictionary: open any entity for its full detail (fields, history, a focused relevance search over what's connected to it), edit a timeline entry in place, or delete something with its references cleaned up automatically.
- **Command line, if you'd rather** — the same chat flow, plus a dedicated tool for editing existing entities directly.
- **Flag it for later** — mark something surfaced during a review as worth a second look, without blocking or changing anything; the flag clears itself once that entity's actually fixed.

## Getting Started

### Requirements

- Python 3.12+
- An OpenAI API key (only needed for the inference/cross-check layers — schema, storage, and the deterministic checks run with no key at all)

### Install

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### Seed a sample world

```bash
python scripts/seed_db.py
```

## Usage

### Web GUI

```bash
streamlit run app.py
```

The sidebar has a live search box (works no matter which tab is active) and two working modes: **Chat** — the same natural-language pipeline as the CLI, rendered as widgets instead of prompts — and **Dictionary** — browse every category, open any entity for its detail view, edit a timeline entry in place, or delete something. A third tab (visualization) is a placeholder for now.

### Command line

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

A new entity tag pauses for a category confirmation and any required fields before reaching the same review step. Type `종료` to exit.

For editing an entity that already exists, `python src/detail_panel.py` looks it up by name or ID, lets you pick a field and enter a new value, and lists every event it's connected to before saving.

## Project Structure

```
schema_registry.yaml       # entity categories, fields, and types — the single source of truth
status_effects.yaml        # reversible statuses and relationship types (imprisoned, exiled, ...)
db/                         # seed data (Markdown with YAML frontmatter) + generated SQLite/Chroma stores
scripts/seed_db.py          # loads db/*.md into SQLite + Chroma
app.py                      # Streamlit GUI entry point

src/
  config.py                  # model-tier -> concrete model name mapping
  schema.py                   # schema_registry.yaml / status_effects.yaml loader + lookup helpers
  storage.py                   # SQLite + Chroma storage layer
  hard_check.py                 # deterministic lifecycle/lifespan checks
  parser.py                      # regex-based input parsing
  mapping.py                      # entity tag resolution + new-entity creation
  inference.py                     # LLM event/attribute inference
  rag_check.py                      # LLM-assisted world-rule / notes / status cross-checks
  archivist.py                       # turns inference + checks into a reviewable diff
  pipeline_session.py                 # pipeline state machine (pause/resume, shared by GUI and CLI)
  deletion.py                          # entity/event deletion with pointer cleanup
  main.py                               # CLI entry point
  field_update.py                       # existing-entity field update logic
  detail_panel.py                        # CLI entity-editing entry point
  flags.py                                # "review later" bookkeeping

tests/                         # pytest suite (many tests call the OpenAI API)
```

## Testing

```bash
pytest
```

Tests covering the LLM-backed layers require `OPENAI_API_KEY` and make real API calls — run a targeted file or test rather than the whole suite by default. The deterministic layers (schema, storage, hard checks, diff assembly, flags) don't need a key.
