from datetime import datetime, timezone

import pytest
import pytest_asyncio

from chatbot.core.config import GraphConfig
from chatbot.persistence.runtime import open_persistence
from chatbot.persistence.threads import ThreadRepository


FIXED_NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def store(tmp_path):
    config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )
    async with open_persistence(config) as handles:
        yield handles.store


@pytest.mark.asyncio
async def test_thread_repository_isolates_clients(store):
    repo = ThreadRepository(store, now=lambda: FIXED_NOW)
    created = await repo.create("client-a", title="新对话")

    assert await repo.owns("client-a", created.thread_id) is True
    assert await repo.owns("client-b", created.thread_id) is False
    assert [item.thread_id for item in await repo.list("client-a")] == [created.thread_id]


@pytest.mark.asyncio
async def test_thread_repository_touch_updates_title_and_orders_newest_first(store):
    moments = iter((
        datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 10, 2, tzinfo=timezone.utc),
    ))
    repo = ThreadRepository(store, now=lambda: next(moments))
    first = await repo.create("client-a", title="第一个")
    second = await repo.create("client-a", title="第二个")
    await repo.touch("client-a", first.thread_id, title="已更新")

    records = await repo.list("client-a")

    assert [(item.thread_id, item.title) for item in records] == [
        (first.thread_id, "已更新"),
        (second.thread_id, "第二个"),
    ]


@pytest.mark.asyncio
async def test_thread_repository_deletes_only_the_client_record(store):
    repo = ThreadRepository(store, now=lambda: FIXED_NOW)
    first = await repo.create("client-a")
    second = await repo.create("client-b")

    await repo.delete_record("client-a", first.thread_id)

    assert await repo.owns("client-a", first.thread_id) is False
    assert await repo.owns("client-b", second.thread_id) is True


@pytest.mark.asyncio
async def test_thread_repository_lists_every_page(store):
    """Catches thread directories silently truncating after the first Store page."""
    repo = ThreadRepository(store, now=lambda: FIXED_NOW)
    for index in range(1_005):
        await repo.create("client-a", title=f"线程 {index}")

    records = await repo.list("client-a")

    assert len(records) == 1_005
    assert {record.title for record in records} == {
        f"线程 {index}" for index in range(1_005)
    }


@pytest.mark.asyncio
async def test_thread_repository_reconciles_stale_records_with_async_exists(store):
    """Catches Saver-missing directory records surviving reconciliation."""
    repo = ThreadRepository(store, now=lambda: FIXED_NOW)
    active = await repo.create("client-a", title="active")
    stale = await repo.create("client-a", title="stale")
    checked: list[str] = []

    async def checkpoint_exists(thread_id: str) -> bool:
        checked.append(thread_id)
        return thread_id == active.thread_id

    records = await repo.list("client-a", exists=checkpoint_exists)

    assert [record.thread_id for record in records] == [active.thread_id]
    assert set(checked) == {active.thread_id, stale.thread_id}
    assert await repo.owns("client-a", stale.thread_id) is False
