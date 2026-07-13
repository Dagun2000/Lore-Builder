**Language:** English | [한국어](README.ko.md)

# Lore Builder

*A consistency-checking pipeline for fantasy worldbuilding.*

Lore Builder takes a single line describing something that just happened in your world — *"In 2100, [Jang] got into a brawl at [the Inn]"* — and checks it against everything already recorded before it's allowed into the database. Hard contradictions (a dead character reappearing, a destroyed item resurfacing) are rejected outright. Anything murkier — a possible world-rule violation, a clash with a character's established traits — is surfaced for a human to decide.

It keeps two kinds of knowledge in sync: structured facts in SQLite (who's related to whom, when things happened, what's currently true) and free-form descriptions in a Chroma vector store, so related lore can be found by meaning rather than by exact ID.

## Features

- **Natural-language event ingestion** — describe what happened in plain text with `[bracketed]` entity tags; the pipeline resolves each tag to an existing entity or walks you through creating a new one.
- **Deterministic hard checks** — lifecycle consistency (no reappearing after your death year, no reappearing after being destroyed) and race-lifespan cross-checks, with zero LLM involvement.
- **LLM-assisted relationship & event inference** — turns a sentence into structured relationships and, where relevant, reversible status changes (imprisoned, cursed, missing, etc.), anchored strictly to entities that already exist.
- **RAG cross-checks** — flags likely world-rule violations, contradictions with a character's established notes, and inconsistencies with an entity's current status, always with a human-readable reason and never applied automatically.
- **Review-before-write** — every change is assembled into a diff and shown item by item before anything touches storage.
- **Existing-entity editing** — a separate CLI for filling in or correcting a field on an entity that already exists, with every related record searched and shown so you can sanity-check before saving.
- **Flag for later** — mark a related record as needing a closer look without blocking or auto-fixing anything.

## Requirements

- Python 3.12+
- An OpenAI API key (only needed for the inference/cross-check layers — schema, storage, and the deterministic hard checks run with no key at all)

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install langchain-openai chromadb pyyaml python-dotenv pytest
```

## Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Record a new event

```bash
python src/main.py
```

```
입력> 2100년, [쟝]이 [주점]에서 술을 마셨다.
[쟝]을(를) char_jang로 인식했습니다.
[주점]을(를) loc_black_goat_inn로 인식했습니다.
[1/2] CREATE timeline: event_...
  승인하시겠습니까? (y/n): y
[2/2] CREATE relationship: rel_...
  승인하시겠습니까? (y/n): y
저장 완료: event_..., rel_...
```

Type `종료` to exit.

### Edit an existing entity

```bash
python src/detail_panel.py
```

Look up an entity by name or ID, pick a field, and enter a new value — related records are searched and shown before anything is saved. Type `목록` at any prompt to see everything flagged for later review.

### Seed the sample world

```bash
python scripts/seed_db.py
```

## Project Structure

```
schema_registry.yaml       # entity categories, fields, and types — the single source of truth
status_effects.yaml        # the fixed set of reversible statuses (imprisoned, cursed, ...)
db/                         # seed data (Markdown with YAML frontmatter) + generated SQLite/Chroma stores
scripts/seed_db.py          # loads db/*.md into SQLite + Chroma

src/
  schema.py                  # schema_registry.yaml loader + lookup helpers
  storage.py                  # SQLite + Chroma storage layer
  hard_check.py                # deterministic lifecycle/lifespan checks
  parser.py                     # regex-based input parsing
  mapping.py                     # entity tag resolution + new-entity creation
  inference.py                    # LLM relationship/event inference
  rag_check.py                     # LLM-assisted world-rule / notes / status cross-checks
  archivist.py                      # turns inference + checks into a reviewable diff
  approval.py                        # the review loop (accept / warn / reject)
  main.py                              # event pipeline entry point
  field_update.py                      # existing-entity field update logic
  detail_panel.py                       # entity-editing entry point
  flags.py                               # "review later" bookkeeping

tests/                         # pytest suite (some tests call the OpenAI API)
```

## Testing

```bash
pytest
```

Tests covering the LLM-backed layers require `OPENAI_API_KEY`; the deterministic layers (schema, storage, hard checks, diff assembly, flags) do not.
