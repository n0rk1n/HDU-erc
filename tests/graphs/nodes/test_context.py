import logging
from types import SimpleNamespace

import pytest

from chatbot.graphs.nodes.context import load_context
from chatbot.memory import MemoryCandidate


@pytest.mark.asyncio
async def test_load_context_reads_profile_and_memory_for_runtime_client_only(
    deps, runtime, store, writer
):
    """Catches profile or memory context crossing the runtime client namespace."""
    await store.aput(("client-a", "profile"), "current", {"response_style": "简短"})
    await store.aput(("client-b", "profile"), "current", {"response_style": "详细"})
    await deps.memory_repository.aremember(
        "client-a", [MemoryCandidate("用户希望继续使用中文。", "preference")]
    )
    await deps.memory_repository.aremember(
        "client-b", [MemoryCandidate("用户希望继续使用英文。", "preference")]
    )

    update = await load_context(
        {"input_message": "继续使用中文", "recent_emotions": []},
        runtime,
        writer,
        deps=deps,
    )

    assert "简短" in update["profile_context"]
    assert "详细" not in update["profile_context"]
    assert "继续使用中文" in update["memory_context"]
    assert "英文" not in update["memory_context"]
    assert isinstance(update["profile_context"], str)
    assert isinstance(update["memory_context"], str)
    assert writer.events == []


@pytest.mark.asyncio
async def test_load_context_uses_emotion_in_memory_query(deps, runtime, writer):
    """Catches context retrieval discarding the current emotional search signal."""
    await deps.memory_repository.aremember(
        "client-a", [MemoryCandidate("用户 anxious 时希望先被倾听。", "preference")]
    )

    update = await load_context(
        {
            "input_message": "继续",
            "emotion_state": {"primary_emotion": "anxious", "confidence": 0.9},
            "recent_emotions": ["anxious"],
        },
        runtime,
        writer,
        deps=deps,
    )

    assert "anxious 时希望先被倾听" in update["memory_context"]


@pytest.mark.asyncio
async def test_load_context_degrades_malformed_store_records_to_empty_strings(
    deps, store, writer, caplog
):
    """Catches Store read failures escaping the node or placing failure objects in state."""
    await store.aput(("client-a", "profile"), "current", ["malformed"])
    await store.aput(("client-a", "memories"), "broken", {"status": "active"})
    runtime = SimpleNamespace(
        context={"client_id": "client-a", "request_id": "req-1", "locale": "zh-CN"},
        store=store,
    )

    with caplog.at_level(logging.WARNING):
        update = await load_context(
            {"input_message": "继续", "recent_emotions": []},
            runtime,
            writer,
            deps=deps,
        )

    assert update == {"profile_context": "", "memory_context": ""}
    assert "profile context read failed" in caplog.text
    assert "memory context read failed" in caplog.text
    assert writer.events == []
