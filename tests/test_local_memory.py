import sqlite3

from chatbot.local_memory import SQLiteLocalMemoryProvider, build_memory_provider
from chatbot.memory import DisabledMemoryProvider, MemoryCandidate, MemoryRuntimeConfig


def test_provider_creates_schema(tmp_path):
    db_path = tmp_path / "memory.sqlite3"

    SQLiteLocalMemoryProvider(str(db_path))

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "select name from sqlite_master where type='table' and name='memories'"
        ).fetchall()
    assert rows == [("memories",)]


def test_remember_inserts_and_search_finds_memory(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    stored = provider.remember([
        MemoryCandidate(
            content="用户希望回答使用中文。",
            category="preference",
            confidence=0.9,
        )
    ])
    results = provider.search("中文回答", limit=5)

    assert len(stored) == 1
    assert len(results) == 1
    assert results[0].content == "用户希望回答使用中文。"
    assert results[0].category == "preference"
    assert results[0].use_count == 1


def test_remember_updates_duplicate_memory(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))
    candidate = MemoryCandidate(
        content="用户希望回答使用中文。",
        category="preference",
        confidence=0.8,
    )

    first = provider.remember([candidate])
    second = provider.remember([
        MemoryCandidate(
            content=" 用户希望回答使用中文。 ",
            category="preference",
            confidence=0.95,
        )
    ])
    all_rows = provider.search("中文", limit=10)

    assert first[0].id == second[0].id
    assert len(all_rows) == 1
    assert all_rows[0].confidence == 0.95


def test_search_returns_empty_for_unmatched_query(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))
    provider.remember([
        MemoryCandidate(content="用户希望回答使用中文。", category="preference")
    ])

    assert provider.search("pizza", limit=5) == []


def test_build_memory_provider_respects_disabled_config(tmp_path):
    provider = build_memory_provider(
        MemoryRuntimeConfig(
            enabled=False,
            db_path=str(tmp_path / "memory.sqlite3"),
            max_results=5,
        )
    )

    assert isinstance(provider, DisabledMemoryProvider)
