import pytest

from src import schema, storage
from scripts.seed_db import seed


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    """Seed data is required by the get_event_years test; upserts make this idempotent."""
    seed()


def test_schema_registry_lifecycle_end_field():
    fields = schema.get_fields_with_role("character", "lifecycle_end")
    names = [f["name"] for f in fields]
    assert names == ["death_year"]


def test_sqlite_save_and_get_roundtrip():
    storage.save_entity(
        "character",
        "char_test_roundtrip",
        {"birth_year": 1999, "race": "race_human", "notes": "test entity"},
    )
    entity = storage.get_entity("character", "char_test_roundtrip")
    assert entity["birth_year"] == 1999
    assert entity["race"] == "race_human"
    assert entity["notes"] == "test entity"


def test_sqlite_upsert_updates_in_place():
    storage.save_entity("character", "char_test_upsert", {"birth_year": 1000})
    storage.save_entity("character", "char_test_upsert", {"birth_year": 2000})

    entity = storage.get_entity("character", "char_test_upsert")
    assert entity["birth_year"] == 2000

    conn = storage.get_connection()
    count = conn.execute(
        'SELECT COUNT(*) FROM "character" WHERE id = ?', ("char_test_upsert",)
    ).fetchone()[0]
    conn.close()
    assert count == 1


def test_chroma_save_and_query():
    storage.save_to_chroma(
        "char_test_chroma",
        "은빛 갑옷을 입은 기사가 용을 무찔렀다는 전설이 전해진다.",
        {"category": "character"},
    )
    results = storage.query_chroma("용을 무찌른 기사 이야기", top_k=3)
    found_ids = results["ids"][0]
    assert "char_test_chroma" in found_ids


def test_get_event_years_for_char_jang():
    assert storage.get_event_years("char_jang") == [2080]
