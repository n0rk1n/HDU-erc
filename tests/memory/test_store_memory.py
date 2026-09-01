from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from langgraph.store.sqlite.aio import AsyncSqliteStore

from chatbot.memory import MemoryCandidate, StoreMemoryRepository


FIXED_NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class MemoryRepoFixture:
    repository: StoreMemoryRepository
    store: AsyncSqliteStore
    client_id: str = "client-a"


@pytest_asyncio.fixture
async def memory_repo(tmp_path):
    async with AsyncSqliteStore.from_conn_string(str(tmp_path / "store.sqlite3")) as store:
        await store.setup()
        yield MemoryRepoFixture(StoreMemoryRepository(store, now=lambda: FIXED_NOW), store)


async def _values(fixture: MemoryRepoFixture, client_id: str = "client-a") -> list[dict]:
    items = await fixture.store.asearch((client_id, "memories"), limit=100)
    return [item.value for item in items]


@pytest.mark.asyncio
async def test_remember_inserts_searches_and_marks_usage(memory_repo):
    stored = await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户希望回答使用中文。", "preference", confidence=0.9)
    ])
    results = await memory_repo.repository.asearch(memory_repo.client_id, "中文回答", limit=5)

    assert len(stored) == 1
    assert [(item.content, item.category, item.use_count) for item in results] == [
        ("用户希望回答使用中文。", "preference", 1)
    ]


@pytest.mark.asyncio
async def test_same_candidate_is_idempotent(memory_repo):
    candidate = MemoryCandidate("用户喜欢简短回答。", "preference")
    first = await memory_repo.repository.aremember(memory_repo.client_id, [candidate])
    second = await memory_repo.repository.aremember(memory_repo.client_id, [candidate])

    assert first[0].id == second[0].id
    assert len(await _values(memory_repo)) == 1


@pytest.mark.asyncio
async def test_memory_is_shared_across_threads_but_isolated_by_client(memory_repo):
    await memory_repo.repository.aremember("client-a", [
        MemoryCandidate("用户喜欢简短回答。", "preference")
    ])

    assert [memory.content for memory in await memory_repo.repository.asearch(
        "client-a", "简短", limit=5
    )] == ["用户喜欢简短回答。"]
    assert await memory_repo.repository.asearch("client-b", "简短", limit=5) == []


@pytest.mark.asyncio
async def test_memory_is_stored_in_the_client_memory_namespace(memory_repo):
    stored = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户喜欢简短回答。", "preference")
    ]))[0]

    assert await memory_repo.store.aget((memory_repo.client_id, "memories"), stored.id) is not None
    assert await memory_repo.store.aget((memory_repo.client_id, "memory_meta"), stored.id) is None


@pytest.mark.asyncio
async def test_invalid_candidates_are_not_stored(memory_repo):
    stored = await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("  ", "preference"),
        MemoryCandidate("用户喜欢简短回答。", "unsupported"),
    ])

    assert stored == []
    assert await _values(memory_repo) == []


@pytest.mark.asyncio
async def test_search_returns_empty_when_limit_is_not_positive(memory_repo):
    await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户希望回答使用中文。", "preference")
    ])

    assert await memory_repo.repository.asearch(memory_repo.client_id, "中文回答", limit=0) == []


@pytest.mark.asyncio
async def test_consolidation_metadata_is_stored_in_its_own_namespace(memory_repo):
    await memory_repo.repository.amark_consolidated(
        memory_repo.client_id, turn_count=5, last_message_id="msg_5", source_checkpoint_id="cp_5"
    )

    assert await memory_repo.store.aget(
        (memory_repo.client_id, "memory_meta"), "consolidation"
    ) is not None
    assert await memory_repo.store.aget(
        (memory_repo.client_id, "memories"), "consolidation"
    ) is None


@pytest.mark.asyncio
async def test_duplicate_normalization_preserves_id_and_merges_maximum_confidence(memory_repo):
    first = await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户希望回答使用中文。", "preference", confidence=0.8)
    ])
    second = await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(" 用户希望回答使用中文。 ", "preference", confidence=0.95)
    ])

    assert first[0].id == second[0].id
    assert [(item.content, item.confidence) for item in await memory_repo.repository.asearch(
        memory_repo.client_id, "中文", limit=10
    )] == [("用户希望回答使用中文。", 0.95)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), [
    ("用户希望回答简洁。", "用户喜欢简洁回答。"),
    ("用户希望回答使用中文。", "用户喜欢用中文回复。"),
])
async def test_similar_preferences_merge(memory_repo, first, second):
    original = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(first, "preference", confidence=0.8)
    ]))[0]
    merged = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(second, "preference", confidence=0.9)
    ]))[0]

    assert merged.id == original.id
    assert merged.content == second
    assert merged.confidence == 0.9


@pytest.mark.asyncio
async def test_conflicting_preferences_supersede_older_memory(memory_repo):
    detailed = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户喜欢详细解释。", "preference")
    ]))[0]
    concise = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户希望回答简洁。", "preference")
    ]))[0]

    assert [item.content for item in await memory_repo.repository.asearch(
        memory_repo.client_id, "回答解释简洁详细", limit=5
    )] == ["用户希望回答简洁。"]
    values = {value["id"]: value for value in await _values(memory_repo)}
    assert values[detailed.id]["status"] == "superseded"
    assert values[concise.id]["supersedes_id"] == detailed.id


@pytest.mark.asyncio
async def test_new_memory_supersedes_every_active_conflict(memory_repo):
    english = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("User prefers replies in English.", "preference")
    ]))[0]
    detailed = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("User prefers detailed answers.", "preference")
    ]))[0]
    replacement = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("User prefers concise replies in Chinese.", "preference")
    ]))[0]

    assert [item.id for item in await memory_repo.repository.asearch(
        memory_repo.client_id, "concise Chinese English detailed replies answers", limit=5
    )] == [replacement.id]
    values = {value["id"]: value for value in await _values(memory_repo)}
    assert values[english.id]["status"] == values[detailed.id]["status"] == "superseded"
    assert values[replacement.id]["supersedes_id"] == english.id


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "first_category", "second", "second_category"), [
    ("User prefers replies in English.", "preference", "User requested not to reply in Chinese.", "boundary"),
    ("User stated that my job is at a third-party hosted platform.", "profile", "User likes third-party hosted services.", "preference"),
    ("User's team writes English responses for support tickets.", "profile", "User prefers replies in Chinese.", "preference"),
    ("User's goal is to draft detailed answers for onboarding docs.", "goal", "User prefers concise replies.", "preference"),
    ("用户喜欢猫。", "preference", "用户喜欢简洁回答。", "preference"),
    ("用户喜欢自然语言处理。", "preference", "用户希望语气自然一点。", "preference"),
    ("User likes Chinese food.", "preference", "User prefers replies in Chinese.", "preference"),
    ("User studies formal methods.", "preference", "User prefers formal tone.", "preference"),
    ("用户喜欢简洁的设计。", "preference", "用户希望回答简洁。", "preference"),
    ("用户学习正式方法。", "preference", "用户希望正式语气。", "preference"),
    ("用户喜欢轻松音乐。", "preference", "用户希望语气轻松。", "preference"),
    ("User likes Chinese food and concise replies.", "preference", "User prefers concise replies in Chinese.", "preference"),
    ("用户喜欢中文歌，也希望回答简洁。", "preference", "用户希望用中文回答，并且回答简洁。", "preference"),
    ("User prefers replies about restaurants in Chinese cities.", "preference", "User prefers replies in Chinese.", "preference"),
    ("用户希望自然一点。", "preference", "用户希望语气自然一点。", "preference"),
    ("User prefers answers about brief history.", "preference", "User prefers brief answers.", "preference"),
])
async def test_unrelated_or_compatible_memories_remain_active(
    memory_repo, first, first_category, second, second_category
):
    initial = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(first, first_category)
    ]))[0]
    next_memory = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(second, second_category)
    ]))[0]

    assert initial.id != next_memory.id
    values = {value["id"]: value for value in await _values(memory_repo)}
    assert values[initial.id]["status"] == values[next_memory.id]["status"] == "active"
    assert values[initial.id]["supersedes_id"] is None
    assert values[next_memory.id]["supersedes_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), [
    ("User requested not to reply in English.", "User prefers replies in English."),
    ("User requires replies in English.", "User prefers concise replies in Chinese."),
    ("用户要求不要使用第三方托管记忆服务。", "用户喜欢方便的第三方托管服务。"),
])
async def test_boundary_conflicts_block_weaker_preferences(memory_repo, first, second):
    protected = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(first, "boundary")
    ]))[0]

    assert await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(second, "preference")
    ]) == []
    assert [value["id"] for value in await _values(memory_repo)] == [protected.id]


@pytest.mark.asyncio
async def test_negated_boundary_supersedes_existing_same_language_preference(memory_repo):
    preference = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("User prefers replies in English.", "preference")
    ]))[0]
    boundary = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("User requested not to reply in English.", "boundary")
    ]))[0]

    assert [item.id for item in await memory_repo.repository.asearch(
        memory_repo.client_id, "English replies", limit=5
    )] == [boundary.id]
    values = {value["id"]: value for value in await _values(memory_repo)}
    assert values[preference.id]["status"] == "superseded"
    assert values[boundary.id]["supersedes_id"] == preference.id


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), [
    ("用户希望回答简洁。", "用户希望回答详细。"),
    ("User prefers concise answers in English.", "User prefers concise answers in Chinese."),
    ("User prefers formal tone.", "User prefers informal tone."),
])
async def test_conflicting_preference_replaces_older_preference(memory_repo, first, second):
    older = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(first, "preference")
    ]))[0]
    newer = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate(second, "preference")
    ]))[0]

    values = {value["id"]: value for value in await _values(memory_repo)}
    assert values[older.id]["status"] == "superseded"
    assert values[newer.id]["status"] == "active"


@pytest.mark.asyncio
async def test_search_ignores_superseded_memory(memory_repo):
    stored = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户希望回答使用中文。", "preference")
    ]))[0]
    item = await memory_repo.store.aget((memory_repo.client_id, "memories"), stored.id)
    await memory_repo.store.aput(
        (memory_repo.client_id, "memories"), stored.id, {**item.value, "status": "superseded"}
    )

    assert await memory_repo.repository.asearch(memory_repo.client_id, "中文回答", limit=5) == []


@pytest.mark.asyncio
async def test_search_tolerates_naive_updated_timestamp(memory_repo):
    stored = (await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户希望回答使用中文。", "preference")
    ]))[0]
    item = await memory_repo.store.aget((memory_repo.client_id, "memories"), stored.id)
    await memory_repo.store.aput(
        (memory_repo.client_id, "memories"), stored.id,
        {**item.value, "updated_at": "2026-06-14T00:00:00"},
    )

    assert [item.id for item in await memory_repo.repository.asearch(
        memory_repo.client_id, "中文回答", limit=5
    )] == [stored.id]


@pytest.mark.asyncio
async def test_search_ranks_boundary_and_exact_phrase_before_other_matches(memory_repo):
    await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户喜欢项目资料里的简洁说明。", "preference", confidence=0.8),
        MemoryCandidate("用户要求在项目资料里标注来源。", "boundary", confidence=0.9),
        MemoryCandidate("User project chatbot answer concise emotion recognition.", "profile"),
        MemoryCandidate("chatbot answer concise", "preference"),
    ])

    chinese = await memory_repo.repository.asearch(memory_repo.client_id, "项目资料说明来源", limit=5)
    english = await memory_repo.repository.asearch(memory_repo.client_id, "chatbot answer concise", limit=5)

    assert chinese[0].category == "boundary"
    assert english[0].content == "chatbot answer concise"


@pytest.mark.asyncio
async def test_search_returns_empty_for_unmatched_or_empty_query(memory_repo):
    await memory_repo.repository.aremember(memory_repo.client_id, [
        MemoryCandidate("用户希望回答使用中文。", "preference")
    ])

    assert await memory_repo.repository.asearch(memory_repo.client_id, "pizza", limit=5) == []
    assert await memory_repo.repository.asearch(memory_repo.client_id, "", limit=5) == []


@pytest.mark.asyncio
async def test_consolidation_state_is_client_scoped_and_records_processed_checkpoints(memory_repo):
    assert await memory_repo.repository.aget_consolidation_state("client-a") == {
        "last_turn_count": 0,
        "last_message_id": None,
        "processed_checkpoint_ids": [],
    }

    await memory_repo.repository.amark_consolidated(
        "client-a", turn_count=5, last_message_id="msg_5", source_checkpoint_id="cp_5"
    )
    await memory_repo.repository.amark_consolidated(
        "client-a", turn_count=8, last_message_id="msg_8", source_checkpoint_id="cp_8"
    )

    assert await memory_repo.repository.aget_consolidation_state("client-a") == {
        "last_turn_count": 8,
        "last_message_id": "msg_8",
        "processed_checkpoint_ids": ["cp_5", "cp_8"],
    }
    client_b_state = await memory_repo.repository.aget_consolidation_state("client-b")
    assert client_b_state["last_turn_count"] == 0
