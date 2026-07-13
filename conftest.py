import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from scripts.seed_db import seed
from src import storage


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    """Reset db/lore.db + chroma_store and reseed, so tests never depend on
    leftover state from a previous pytest run (individual test files must
    be able to run standalone, in any order, repeatedly)."""
    if storage.DB_PATH.exists():
        storage.DB_PATH.unlink()
    if storage.CHROMA_PATH.exists():
        shutil.rmtree(storage.CHROMA_PATH)
    seed()
