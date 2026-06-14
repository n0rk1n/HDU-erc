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


def test_provider_migrates_existing_schema(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            create table memories (
                id text primary key,
                content text not null,
                category text not null,
                source text not null,
                confidence real not null,
                created_at text not null,
                updated_at text not null,
                last_used_at text,
                use_count integer not null default 0
            )
            """
        )
        connection.execute(
            """
            insert into memories (
                id, content, category, source, confidence,
                created_at, updated_at, last_used_at, use_count
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "mem_old",
                "用户希望回答使用中文。",
                "preference",
                "chat",
                0.9,
                "2026-06-14T00:00:00+00:00",
                "2026-06-14T00:00:00+00:00",
                None,
                0,
            ),
        )

    provider = SQLiteLocalMemoryProvider(str(db_path))

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("pragma table_info(memories)").fetchall()
        }
        status = connection.execute(
            "select status from memories where id = ?",
            ("mem_old",),
        ).fetchone()[0]

    assert {"status", "supersedes_id", "metadata_json"} <= columns
    assert status == "active"
    assert provider.search("中文回答", limit=5)[0].id == "mem_old"


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


def test_search_excludes_superseded_memories(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))
    provider.remember([
        MemoryCandidate(content="用户希望回答使用中文。", category="preference")
    ])

    with sqlite3.connect(provider.db_path) as connection:
        connection.execute("update memories set status = 'superseded'")

    assert provider.search("中文回答", limit=5) == []


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
