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


def test_remember_merges_similar_concise_preferences(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户希望回答简洁。", category="preference", confidence=0.8)
    ])
    second = provider.remember([
        MemoryCandidate(content="用户喜欢简洁回答。", category="preference", confidence=0.9)
    ])
    results = provider.search("简洁回答", limit=5)

    assert first[0].id == second[0].id
    assert len(results) == 1
    assert results[0].content == "用户喜欢简洁回答。"
    assert results[0].confidence == 0.9


def test_remember_merges_similar_chinese_reply_preferences(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户希望回答使用中文。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户喜欢用中文回复。", category="preference")
    ])

    assert first[0].id == second[0].id
    assert len(provider.search("中文回复", limit=5)) == 1


def test_remember_does_not_merge_conflicting_known_preferences(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户希望回答简洁。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户希望回答详细。", category="preference")
    ])
    results = provider.search("回答简洁详细", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户希望回答简洁。",
        "用户希望回答详细。",
    }


def test_remember_does_not_merge_mixed_language_preferences(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="User prefers concise answers in English.", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="User prefers concise answers in Chinese.", category="preference")
    ])
    results = provider.search("concise English Chinese", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "User prefers concise answers in English.",
        "User prefers concise answers in Chinese.",
    }


def test_remember_does_not_merge_boilerplate_overlap_with_known_topic(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户喜欢猫。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户喜欢简洁回答。", category="preference")
    ])
    results = provider.search("喜欢猫简洁回答", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户喜欢猫。",
        "用户喜欢简洁回答。",
    }


def test_remember_does_not_merge_formal_and_informal_tone_preferences(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="User prefers formal tone.", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="User prefers informal tone.", category="preference")
    ])
    results = provider.search("formal informal tone", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "User prefers formal tone.",
        "User prefers informal tone.",
    }


def test_remember_does_not_treat_nlp_as_casual_tone(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户喜欢自然语言处理。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户希望语气自然一点。", category="preference")
    ])
    results = provider.search("自然语言处理语气", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户喜欢自然语言处理。",
        "用户希望语气自然一点。",
    }


def test_remember_does_not_treat_chinese_food_as_reply_language(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="User likes Chinese food.", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="User prefers replies in Chinese.", category="preference")
    ])
    results = provider.search("Chinese food replies", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "User likes Chinese food.",
        "User prefers replies in Chinese.",
    }


def test_remember_does_not_treat_formal_methods_as_tone(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="User studies formal methods.", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="User prefers formal tone.", category="preference")
    ])
    results = provider.search("formal methods tone", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "User studies formal methods.",
        "User prefers formal tone.",
    }


def test_remember_does_not_treat_concise_design_as_reply_length(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户喜欢简洁的设计。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户希望回答简洁。", category="preference")
    ])
    results = provider.search("简洁设计回答", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户喜欢简洁的设计。",
        "用户希望回答简洁。",
    }


def test_remember_does_not_treat_formal_methods_as_chinese_tone(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户学习正式方法。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户希望正式语气。", category="preference")
    ])
    results = provider.search("正式方法语气", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户学习正式方法。",
        "用户希望正式语气。",
    }


def test_remember_does_not_treat_relaxing_music_as_casual_tone(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户喜欢轻松音乐。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户希望语气轻松。", category="preference")
    ])
    results = provider.search("轻松音乐语气", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户喜欢轻松音乐。",
        "用户希望语气轻松。",
    }


def test_remember_does_not_bind_chinese_food_to_reply_language(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="User likes Chinese food and concise replies.", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="User prefers concise replies in Chinese.", category="preference")
    ])
    results = provider.search("Chinese food concise replies", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "User likes Chinese food and concise replies.",
        "User prefers concise replies in Chinese.",
    }


def test_remember_does_not_bind_chinese_song_to_reply_language(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户喜欢中文歌，也希望回答简洁。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户希望用中文回答，并且回答简洁。", category="preference")
    ])
    results = provider.search("中文歌回答简洁", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户喜欢中文歌，也希望回答简洁。",
        "用户希望用中文回答，并且回答简洁。",
    }


def test_remember_does_not_bind_chinese_cities_to_reply_language(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="User prefers replies about restaurants in Chinese cities.", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="User prefers replies in Chinese.", category="preference")
    ])
    results = provider.search("replies restaurants Chinese cities", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "User prefers replies about restaurants in Chinese cities.",
        "User prefers replies in Chinese.",
    }


def test_remember_does_not_treat_bare_natural_as_casual_tone(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="用户希望自然一点。", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="用户希望语气自然一点。", category="preference")
    ])
    results = provider.search("自然一点语气", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "用户希望自然一点。",
        "用户希望语气自然一点。",
    }


def test_remember_does_not_bind_brief_history_to_reply_length(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    first = provider.remember([
        MemoryCandidate(content="User prefers answers about brief history.", category="preference")
    ])
    second = provider.remember([
        MemoryCandidate(content="User prefers brief answers.", category="preference")
    ])
    results = provider.search("answers brief history", limit=5)

    assert first[0].id != second[0].id
    assert len(results) == 2
    assert {memory.content for memory in results} == {
        "User prefers answers about brief history.",
        "User prefers brief answers.",
    }


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
