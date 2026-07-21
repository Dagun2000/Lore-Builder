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

Nothing about the schema, the checks, or the prompts assumes any particular genre or setting — the sample world here happens to be a fantasy one, but a sci-fi, contemporary, or historical setting works the same way. See [Customizing](#customizing) for what to change if you want the AI's own phrasing to reflect that.

## Features

- **Natural-language entry** — describe events and facts in plain text, tagging entities with `[brackets]`. An unrecognized tag walks you through creating a new entity: category, required fields, and as many optional details as can be inferred from context.
- **A timeline, not a wiki page** — every event, relationship, and reversible status (imprisoned, cursed, exiled, ...) is one entry in a shared timeline, linked to everyone it involves. Nothing stores a "current state" snapshot — status is always computed from the timeline itself, since the world has no fixed sense of "now": history can be added in any year, in any order.
- **Deterministic checks** — lifecycle consistency (no reappearing after a death or destruction date) and race/lifespan cross-checks, resolved with plain logic, no model call needed.
- **LLM-assisted understanding** — turns a sentence into structured facts (who did what, to whom, when), and, for a new entity, separates out lifecycle attributes, persistent traits, and any event still worth recording in its own right.
- **Contradiction checks on everything new** — flags likely rule violations and clashes with an entity's own established notes: explicit constraints, tone mismatches (a "life-threatening" ruin and a carefree picnic), even a number that doesn't hold up against a stated correlation ("more means stronger," but this one barely qualifies) — while still honoring any exception the rule itself allows for. This runs for a dated event and for a bare new-entity introduction alike, always with a reason and a chance to override.
- **One decision per change** — a change is bundled into a single diff (the core fact plus whatever else needs updating alongside it) and shown as one approve/reject choice, not one prompt per row touched.
- **Registries you control** — the set of reversible statuses and relationship types (imprisoned, exiled, allied-with, ...) is a plain editable list, not hardcoded, and each entry carries a short note on what it actually means so the AI doesn't have to guess from the name alone; a genuinely new type Step 3 needs gets proposed to you for approval before it's added.
- **Creator Mode** — describe a whole story beat instead of one fact at a time; the AI drafts every event it implies, checks its own draft against the same rules and cross-checks a normal entry goes through, and retries if something doesn't hold up — optionally inventing throwaway supporting characters or items for a scene, if you opt in per category.
- **Visualize any entity** — a timeline swimlane of everything it's been part of, and a relationship graph of everyone and everything one hop away, both generated straight from the data with no extra bookkeeping.
- **Bring your own LLM** — OpenAI, Anthropic (Claude), Google (Gemini), or a local Ollama model, picked with one environment variable — no code changes.
- **Bilingual interface** — toggle the GUI's own menus and messages between Korean and English anytime; entirely separate from what language your world's own content is written in.
- **Web GUI** — chat for the same natural-language flow, plus a browsable dictionary: open any entity for its full detail (fields, history, a focused relevance search over what's connected to it), edit a timeline entry in place, or delete something with its references cleaned up automatically.
- **Command line, if you'd rather** — the same chat flow, plus a dedicated tool for editing existing entities directly.
- **Flag it for later** — mark something surfaced during a review as worth a second look, without blocking or changing anything; the flag clears itself once that entity's actually fixed.

## Getting Started

### Requirements

- Python 3.12+
- An API key for at least one supported LLM provider (OpenAI, Anthropic, or Google) — or a locally running [Ollama](https://ollama.com) instance, which needs no key at all. Schema, storage, and the deterministic checks run with no LLM either way.

### Install

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Copy the example env file and fill in what you need:

```bash
cp .env.example .env
```

At minimum, set `LLM_PROVIDER` (`openai`, `anthropic`, `ollama`, or `google`) and that provider's API key — or, for Ollama, point `OLLAMA_BASE_URL` at a running instance. Everything else in `.env.example` (per-provider model overrides, `WORLD_LANGUAGE`) has a working default; see [Customizing](#customizing) before changing `WORLD_LANGUAGE`.

### Seed a sample world

```bash
python scripts/seed_db.py
```

## Usage

### Web GUI

```bash
streamlit run app.py
```

The sidebar has a language toggle (한국어/English — this only translates the GUI's own menus and messages, not your world's content or what the AI writes), a live search box that works no matter which tab is active, a **🚩 Flags** expander listing anything marked for a later look, and two modes:

- **Chat** — the same natural-language pipeline as the CLI, rendered as widgets instead of prompts. A toggle above the input switches between **Normal chat** (one fact or event at a time) and **Creator Mode** (describe a whole story beat; the AI drafts every event it implies, validates its own draft, and retries on rejection — plus per-category checkboxes to let it invent throwaway supporting entities where you want one).
- **Dictionary** — browse every category, including a **Relations/Status** entry for editing the reversible-status/relationship registry directly (add one, edit its description, or delete one that's not in use). Opening any entity shows its full detail — fields, a focused relevance search, timeline participation — and, for anything other than a raw timeline record, a **Visualization** tab with a timeline swimlane and a relationship graph.

### Command line

```bash
python src/main.py
```

```
Lore Builder CLI. 종료하려면 '종료'를 입력하세요.

입력> 2100년, [미라]가 [은빛도시]에서 산책을 했다.
CREATE timeline: event_2100년_미라가_은빛도시에서_산책했다
  근거: 새 사건 기록: 2100년 미라가 은빛도시에서 산책했다.
  필드: {'year': 2100, 'location': 'loc_은빛도시', 'notes': '2100년 미라가 은빛도시에서 산책했다.'}
  함께 갱신되는 엔티티: char_미라, loc_은빛도시
저장하시겠습니까? [저장/취소]: 저장
저장 완료: event_2100년_미라가_은빛도시에서_산책했다, char_미라(갱신), loc_은빛도시(갱신)
```

A new entity tag pauses for a category confirmation and any required fields before reaching the same review step. Type `종료` to exit.

For editing an entity that already exists, `python src/detail_panel.py` looks it up by name or ID, lets you pick a field and enter a new value, and lists every event it's connected to before saving.

## Customizing

Everything below is a config file or a plain-text prompt, not a code change — the intent throughout this project is that a different world, a different genre, or a different model provider is a matter of editing data, not logic.

### Adding or changing entity categories

`schema_registry.yaml` is the single source of truth for what categories exist and what fields each one has. A category looks like:

```yaml
faction:
  id_prefix: faction_
  fields:
    - name: name
      type: text
      required: true
    - name: category
      type: enum
      options: [mercenary_guild, kingdom, religious_order, tribe]
      required: true
    - name: founded_year
      type: integer
      required: false
      role: lifecycle_start
    - name: disbanded_year
      type: integer
      required: false
      role: lifecycle_end
    - name: notes
      type: text
      required: false
    - name: event_ids
      type: list
      required: false
```

Field types: `text`, `integer`, `boolean`, `enum` (with an `options` list), `reference` (points at another category via `ref_category`, or `any` to allow every category), and `list`. The `role: lifecycle_start` / `lifecycle_end` tags mark which field acts as that category's "birth"/"death" year for the deterministic checks and for computing current state — give any new category with a lifespan the same pair. Add a category (or a field to an existing one) and it shows up everywhere automatically: the Dictionary picker, entity creation, tag resolution, reference dropdowns — nothing else needs to know about it by name.

### Reversible statuses and relationships

`status_effects.yaml` is the registry of every reversible personal status (imprisoned, cursed, ...) and relationship type (allied-with, enemies-with, ...):

```yaml
- id: imprisoned
  label: 수감
  type: individual
  notes: 물리적으로 수감 장소를 벗어난 행동(다른 지역 방문, 자유로운 이동 등)은 불가능하다. 감방 안에서의 행동(대화, 생각, 식사 등)은 가능하다.
```

`type` is `individual` (a status with no target, like *cursed*) or `relational` (targets another entity, like *exiled from* or *allied with*). `notes` is free text describing what the status actually implies — it's fed directly into every relevant LLM check, so the more concrete it is about what is and isn't possible while active, the better the checks get. Edit the file directly, or from the GUI: **Dictionary → Relations/Status**, which lets you add an entry, edit its notes, or delete one (blocked with a warning if anything currently uses it). A genuinely new relationship type that comes up mid-conversation is proposed to you before it's added to the registry permanently — you're never surprised by a new entry appearing on its own.

### Editing the seed world

`db/{category}/*.md` is the starting data any fresh `scripts/seed_db.py` run loads. Each file is a YAML frontmatter block (an `id` plus whatever fields that category defines) followed by free-form body text used for relevance search:

```
---
id: char_쟝
name: 쟝
...
event_ids:
  - event_쟝_2080
---

쟝(Jang)은 용병 길드 소속의 인간 용병이다. ...
```

A few things to keep in mind when hand-editing these:

- **Pointers are manual and bidirectional.** If you add a timeline event that references a character, also add that event's id to the character's own `event_ids` list (and to any other entity it involves) — nothing auto-syncs a hand-edited file the way saving through the app does.
- **Give an id the same script as the entity's own name.** The sample data's ids are Korean slugs of the entity's name (`char_쟝`, not `char_jang`) — consistent with what the app itself generates for anything created through the pipeline. Mixing scripts (a Korean-named entity with a romanized id) is exactly what once caused the AI to reconstruct a guessed, garbled rendering of the name from the id instead of using the real one — keep new ids in whatever script your `name` field is in.
- **Reseed after editing.** Delete `db/lore.db` and the `chroma_store/` directory, then run `python scripts/seed_db.py` again — seeding doesn't retroactively clean up rows for files you removed.

### Choosing an LLM provider

Set `LLM_PROVIDER` in `.env` to `openai`, `anthropic`, `ollama`, or `google`; `.env.example` lists every provider's API key variable and optional per-tier model overrides (a cheaper "simple" model and a stronger "reasoning" model). Ollama needs no API key, just `OLLAMA_BASE_URL` pointing at a running instance.

`PARALLEL_RAG_CHECKS` (defaults to `true`) controls whether the pipeline's two independent per-event checks fire concurrently — a real latency win on a cloud provider, but set it to `false` if you're running a local Ollama model sized to fill your available VRAM, where two concurrent generations can contend for the same GPU memory instead of actually running in parallel.

### Setting the world language

`WORLD_LANGUAGE` in `.env` (defaults to `ko`, matching the sample world) declares what language your own world's content — and the enum options you write in `schema_registry.yaml`/`status_effects.yaml` — is in. **Set it once, before entering any data.** Changing it mid-project risks orphaning already-saved values against a since-changed options list and mixing two languages into one project's lore. As of now this is a declared setting, not yet an enforced one — the AI's own prompts are still fixed in Korean regardless of this value; see the next section if you want to change that yourself.

### Interface language

The sidebar's 한국어/English toggle only affects the GUI's own labels, buttons, and messages — it's independent of `WORLD_LANGUAGE` and never touches what you type, what gets saved, or what the AI writes back.

### Adapting the AI's prompts to your own setting

The schema and checks don't assume a genre, but the LLM prompts do open with a short role description that currently says "판타지 세계관" ("a fantasy-world lore database") — search for that phrase across `src/creator.py`, `src/field_update.py`, `src/inference.py`, `src/mapping.py`, and `src/rag_check.py` and edit it to describe your own setting (sci-fi, contemporary, historical, or anything else) if you'd like the AI's framing to match. Nothing else about the prompts, schema, or checks needs to change for a different genre.

## Project Structure

```
schema_registry.yaml       # entity categories, fields, and types — the single source of truth
status_effects.yaml        # reversible statuses and relationship types (imprisoned, exiled, ...)
.env.example                # LLM provider + world-language config template
db/                         # seed data (Markdown with YAML frontmatter) + generated SQLite/Chroma stores
scripts/seed_db.py          # loads db/*.md into SQLite + Chroma
app.py                      # Streamlit GUI entry point

src/
  config.py                  # LLM_PROVIDER -> concrete chat-model factory (OpenAI/Anthropic/Ollama/Google)
  settings.py                  # WORLD_LANGUAGE reader
  i18n.py                       # interface-language (한국어/English) translation registry
  schema.py                      # schema_registry.yaml / status_effects.yaml loader + lookup helpers
  storage.py                      # SQLite + Chroma storage layer
  hard_check.py                    # deterministic lifecycle/lifespan checks
  parser.py                         # regex-based input parsing
  mapping.py                         # entity tag resolution + new-entity creation
  inference.py                        # LLM event/attribute inference
  rag_check.py                         # LLM-assisted world-rule / notes / status cross-checks
  archivist.py                          # turns inference + checks into a reviewable diff
  pipeline_session.py                    # pipeline state machine (pause/resume, shared by GUI and CLI)
  creator.py                              # Creator Mode: drafts + validates a multi-event narrative
  creator_session.py                       # Creator Mode's own pause/resume state machine
  visualization.py                          # timeline + relationship-graph data prep for the GUI
  deletion.py                                # entity/event deletion with pointer cleanup
  main.py                                     # CLI entry point
  approval.py                                  # CLI approval/review prompts
  field_update.py                               # existing-entity field update logic
  detail_panel.py                                # CLI entity-editing entry point
  flags.py                                        # "review later" bookkeeping

tests/                         # pytest suite (many tests call an LLM provider)
```

## Testing

```bash
pytest
```

Tests covering the LLM-backed layers make real API calls against whatever `LLM_PROVIDER` is configured — run a targeted file or test rather than the whole suite by default. The deterministic layers (schema, storage, hard checks, diff assembly, flags) don't need a provider configured at all.
