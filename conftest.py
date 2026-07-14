import shutil
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE_DIR))

import pytest

from scripts.seed_db import seed
from src import storage

# Tests get their own SQLite file + Chroma store, entirely separate from
# db/lore.db + chroma_store/ (what `streamlit run app.py` / the CLI read).
# Without this, running the suite while the GUI is in use — or just having
# run it once before — leaves test fixtures (char_e2e_*, char_p8_*, etc.)
# sitting in the real dev/demo data, visible in the GUI. This reassignment
# must happen before anything opens a SQLite connection or a Chroma client
# (the client in particular is cached at module scope in storage.py), so it
# runs at import time here, not inside the fixture below.
storage.DB_PATH = _BASE_DIR / "db" / "test_lore.db"
storage.CHROMA_PATH = _BASE_DIR / "chroma_store_test"


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    """Reset the test DB/Chroma store and reseed, so tests never depend on
    leftover state from a previous pytest run (individual test files must
    be able to run standalone, in any order, repeatedly)."""
    if storage.DB_PATH.exists():
        storage.DB_PATH.unlink()
    if storage.CHROMA_PATH.exists():
        shutil.rmtree(storage.CHROMA_PATH)
    seed()
