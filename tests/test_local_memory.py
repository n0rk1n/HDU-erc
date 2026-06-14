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
    assert [memory.content for memory in results] == ["用户希望回答详细。"]


def test_conflicting_preferences_supersede_older_memory(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    detailed = provider.remember([
        MemoryCandidate(content="用户喜欢详细解释。", category="preference")
    ])[0]
    concise = provider.remember([
        MemoryCandidate(content="用户希望回答简洁。", category="preference")
    ])[0]
    results = provider.search("回答解释简洁详细", limit=5)

    assert concise.id != detailed.id
    assert [memory.content for memory in results] == ["用户希望回答简洁。"]
    with sqlite3.connect(provider.db_path) as connection:
        status = connection.execute(
            "select status from memories where id = ?",
            (detailed.id,),
        ).fetchone()[0]
        supersedes_id = connection.execute(
            "select supersedes_id from memories where id = ?",
            (concise.id,),
        ).fetchone()[0]
    assert status == "superseded"
    assert supersedes_id == detailed.id


def test_boundary_is_not_superseded_by_weaker_preference(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    boundary = provider.remember([
        MemoryCandidate(
            content="用户要求不要使用第三方托管记忆服务。",
            category="boundary",
            confidence=0.9,
        )
    ])[0]
    provider.remember([
        MemoryCandidate(
            content="用户喜欢方便的第三方托管服务。",
            category="preference",
            confidence=0.8,
        )
    ])
    results = provider.search("第三方托管记忆服务", limit=5)

    assert any(memory.id == boundary.id for memory in results)
    with sqlite3.connect(provider.db_path) as connection:
        status = connection.execute(
            "select status from memories where id = ?",
            (boundary.id,),
        ).fetchone()[0]
    assert status == "active"


def test_negated_boundary_language_does_not_supersede_compatible_preference(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    english = provider.remember([
        MemoryCandidate(
            content="User prefers replies in English.",
            category="preference",
        )
    ])[0]
    boundary = provider.remember([
        MemoryCandidate(
            content="User requested not to reply in Chinese.",
            category="boundary",
        )
    ])[0]
    results = provider.search("English Chinese replies", limit=5)

    assert any(memory.id == english.id for memory in results)
    assert any(memory.id == boundary.id for memory in results)
    with sqlite3.connect(provider.db_path) as connection:
        english_status = connection.execute(
            "select status from memories where id = ?",
            (english.id,),
        ).fetchone()[0]
    assert english_status == "active"


def test_negated_same_language_boundary_blocks_weaker_preference(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    boundary = provider.remember([
        MemoryCandidate(
            content="User requested not to reply in English.",
            category="boundary",
        )
    ])[0]
    stored = provider.remember([
        MemoryCandidate(
            content="User prefers replies in English.",
            category="preference",
        )
    ])
    results = provider.search("English replies", limit=5)

    assert stored == []
    assert [memory.id for memory in results] == [boundary.id]
    with sqlite3.connect(provider.db_path) as connection:
        boundary_status = connection.execute(
            "select status from memories where id = ?",
            (boundary.id,),
        ).fetchone()[0]
        preference_count = connection.execute(
            "select count(*) from memories where content = ?",
            ("User prefers replies in English.",),
        ).fetchone()[0]
    assert boundary_status == "active"
    assert preference_count == 0


def test_negated_same_language_boundary_supersedes_existing_preference(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    preference = provider.remember([
        MemoryCandidate(
            content="User prefers replies in English.",
            category="preference",
        )
    ])[0]
    boundary = provider.remember([
        MemoryCandidate(
            content="User requested not to reply in English.",
            category="boundary",
        )
    ])[0]
    results = provider.search("English replies", limit=5)

    assert [memory.id for memory in results] == [boundary.id]
    with sqlite3.connect(provider.db_path) as connection:
        statuses = dict(
            connection.execute(
                "select id, status from memories where id in (?, ?)",
                (preference.id, boundary.id),
            ).fetchall()
        )
        supersedes_id = connection.execute(
            "select supersedes_id from memories where id = ?",
            (boundary.id,),
        ).fetchone()[0]
    assert statuses == {
        preference.id: "superseded",
        boundary.id: "active",
    }
    assert supersedes_id == preference.id


def test_hosted_profile_fact_does_not_conflict_with_hosted_preference(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    profile = provider.remember([
        MemoryCandidate(
            content="User stated that my job is at a third-party hosted platform.",
            category="profile",
        )
    ])[0]
    preference = provider.remember([
        MemoryCandidate(
            content="User likes third-party hosted services.",
            category="preference",
        )
    ])[0]
    results = provider.search("third-party hosted platform services", limit=5)

    assert any(memory.id == profile.id for memory in results)
    assert any(memory.id == preference.id for memory in results)
    with sqlite3.connect(provider.db_path) as connection:
        profile_status = connection.execute(
            "select status from memories where id = ?",
            (profile.id,),
        ).fetchone()[0]
    assert profile_status == "active"


def test_new_memory_supersedes_all_active_conflicts(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    english = provider.remember([
        MemoryCandidate(content="User prefers replies in English.", category="preference")
    ])[0]
    detailed = provider.remember([
        MemoryCandidate(content="User prefers detailed answers.", category="preference")
    ])[0]
    concise_chinese = provider.remember([
        MemoryCandidate(
            content="User prefers concise replies in Chinese.",
            category="preference",
        )
    ])[0]
    results = provider.search("concise Chinese English detailed replies answers", limit=5)

    assert [memory.id for memory in results] == [concise_chinese.id]
    with sqlite3.connect(provider.db_path) as connection:
        statuses = dict(
            connection.execute(
                "select id, status from memories where id in (?, ?, ?)",
                (english.id, detailed.id, concise_chinese.id),
            ).fetchall()
        )
        supersedes_id = connection.execute(
            "select supersedes_id from memories where id = ?",
            (concise_chinese.id,),
        ).fetchone()[0]
    assert statuses == {
        english.id: "superseded",
        detailed.id: "superseded",
        concise_chinese.id: "active",
    }
    assert supersedes_id == english.id


def test_mixed_boundary_conflict_blocks_candidate_without_superseding(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))

    boundary = provider.remember([
        MemoryCandidate(
            content="User requires replies in English.",
            category="boundary",
        )
    ])[0]
    detailed = provider.remember([
        MemoryCandidate(
            content="User prefers detailed answers.",
            category="preference",
        )
    ])[0]

    stored = provider.remember([
        MemoryCandidate(
            content="User prefers concise replies in Chinese.",
            category="preference",
        )
    ])
    results = provider.search("English detailed concise Chinese replies answers", limit=5)

    assert stored == []
    assert {memory.id for memory in results} == {boundary.id, detailed.id}
    with sqlite3.connect(provider.db_path) as connection:
        statuses = dict(
            connection.execute(
                "select id, status from memories where id in (?, ?)",
                (boundary.id, detailed.id),
            ).fetchall()
        )
        candidate_count = connection.execute(
            "select count(*) from memories where content = ?",
            ("User prefers concise replies in Chinese.",),
        ).fetchone()[0]
    assert statuses == {
        boundary.id: "active",
        detailed.id: "active",
    }
    assert candidate_count == 0


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
    assert [memory.content for memory in results] == [
        "User prefers concise answers in Chinese."
    ]


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
    assert [memory.content for memory in results] == ["User prefers informal tone."]


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


def test_search_tolerates_offset_naive_updated_at(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))
    stored = provider.remember([
        MemoryCandidate(content="用户希望回答使用中文。", category="preference")
    ])[0]

    with sqlite3.connect(provider.db_path) as connection:
        connection.execute(
            "update memories set updated_at = ? where id = ?",
            ("2026-06-14T00:00:00", stored.id),
        )

    results = provider.search("中文回答", limit=5)

    assert [memory.id for memory in results] == [stored.id]


def test_search_prioritizes_boundary_over_preference(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))
    provider.remember([
        MemoryCandidate(
            content="用户喜欢项目资料里的简洁说明。",
            category="preference",
            confidence=0.8,
        ),
        MemoryCandidate(
            content="用户要求在项目资料里标注来源。",
            category="boundary",
            confidence=0.9,
        ),
    ])

    results = provider.search("项目资料说明来源", limit=5)

    assert results[0].category == "boundary"


def test_search_prioritizes_more_specific_phrase_match(tmp_path):
    provider = SQLiteLocalMemoryProvider(str(tmp_path / "memory.sqlite3"))
    provider.remember([
        MemoryCandidate(
            content="User project chatbot answer concise emotion recognition.",
            category="profile",
        ),
        MemoryCandidate(content="chatbot answer concise", category="preference"),
    ])

    results = provider.search("chatbot answer concise", limit=5)

    assert results[0].content == "chatbot answer concise"


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
