"""Project-level settings (Phase 10 patch 18 — i18n foundation).

`world_language` is fundamentally different from the interface language
(see i18n.py): it's what the AI reads/writes/reasons in, and the anchor
enum values are stored against — so it's fixed once per project, not a
runtime toggle. Switching it mid-project would either orphan already-saved
data (a stored enum value like character.gender="남성" wouldn't match a
freshly-translated options list) or silently start writing new content in
a different language than everything already in the world, fragmenting a
single project's own lore into a mixed-language mess. Neither is something
a casual toggle should be able to trigger, so it's read from WORLD_LANGUAGE
in .env — the same place LLM_PROVIDER lives (see config.py) — set once
before you start entering data, not exposed anywhere in the GUI.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


def get_world_language() -> str:
    return os.environ.get("WORLD_LANGUAGE", "ko").strip().lower()
