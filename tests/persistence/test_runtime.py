import pytest
from langgraph.checkpoint.base import empty_checkpoint

from chatbot.core.config import GraphConfig
from chatbot.persistence.runtime import open_persistence


@pytest.mark.asyncio
async def test_open_persistence_uses_separate_sqlite_files(tmp_path):
    config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )

    async with open_persistence(config) as handles:
        await handles.store.aput(("client-1", "profile"), "current", {"preferred_name": "小明"})
        item = await handles.store.aget(("client-1", "profile"), "current")

    assert item is not None
    assert item.value["preferred_name"] == "小明"
    assert (tmp_path / "checkpoints.sqlite3").exists()
    assert (tmp_path / "store.sqlite3").exists()


@pytest.mark.asyncio
async def test_open_persistence_rejects_unsafe_checkpoint_values_without_saving(tmp_path):
    class UnsafeCheckpointValue:
        pass

    config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"unsafe": UnsafeCheckpointValue()}
    checkpoint_config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    async with open_persistence(config) as handles:
        with pytest.raises(TypeError):
            await handles.checkpointer.aput(checkpoint_config, checkpoint, {}, {})
        saved = [item async for item in handles.checkpointer.alist(checkpoint_config)]

    assert saved == []
