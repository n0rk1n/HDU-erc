import logging
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from chatbot.graphs.nodes.memory import extract_memory, maybe_consolidate
from chatbot.graphs.state import ConversationState


class FailingMemoryRepository:
    async def aremember(self, client_id, candidates):
        raise RuntimeError("store unavailable")

    async def aget_consolidation_state(self, client_id):
        raise RuntimeError("store unavailable")


def test_compiled_state_retains_memory_warning_transient():
    """Catches the graph schema silently dropping a non-fatal memory warning."""
    builder = StateGraph(ConversationState)
    builder.add_node(
        "warn",
        lambda state: {"memory_warning": "memory_write_failed"},
    )
    builder.add_edge(START, "warn")
    builder.add_edge("warn", END)

    result = builder.compile().invoke({"messages": []})

    assert result["memory_warning"] == "memory_write_failed"


def completed_turn_state(user_message: str, assistant_reply: str) -> dict:
    return {
        "request_id": "req-1",
        "turn_count": 1,
        "input_message": user_message,
        "response_content": assistant_reply,
        "messages": [
            HumanMessage(id="human_req-1", content=user_message),
            AIMessage(id="ai_req-1", content=assistant_reply),
        ],
        "thread_meta": {"thread_id": "thread-a"},
    }


@pytest.mark.asyncio
async def test_extract_memory_is_idempotent_for_same_request(deps, runtime, writer):
    """Catches retrying a node creating duplicate durable memories."""
    state = completed_turn_state("我希望以后都用中文回答。", "好的。")
    config = {"configurable": {"thread_id": "thread-a"}}

    first = await extract_memory(state, runtime, writer, config, deps=deps)
    second = await extract_memory(state, runtime, writer, config, deps=deps)
    memories = await deps.memory_repository.asearch("client-a", "中文", limit=5)

    assert len(memories) == 1
    assert memories[0].content == "用户希望以后都用中文回答。"
    assert first == {"memory_warning": ""}
    assert second == {"memory_warning": ""}
    assert writer.events == []


@pytest.mark.asyncio
async def test_consolidation_persists_checkpoint_id_and_is_idempotent(
    deps, runtime, writer, monkeypatch
):
    """Catches retrying one checkpoint consolidating the same window twice."""
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "true")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_INTERVAL", "5")
    state = {
        "request_id": "req-5",
        "turn_count": 5,
        "messages": [
            HumanMessage(id="human_req-4", content="我只想被倾听，不要急着给建议。"),
            AIMessage(id="ai_req-4", content="好，我先听你说。"),
            HumanMessage(id="human_req-5", content="还是只想被倾听，不要急着给建议。"),
            AIMessage(id="ai_req-5", content="我在听。"),
        ],
        "thread_meta": {"thread_id": "thread-a"},
    }
    config = {
        "configurable": {"thread_id": "thread-a", "checkpoint_id": "checkpoint-5"}
    }

    first = await maybe_consolidate(state, runtime, writer, config, deps=deps)
    second = await maybe_consolidate(state, runtime, writer, config, deps=deps)
    consolidation = await deps.memory_repository.aget_consolidation_state("client-a")
    memories = await deps.memory_repository.asearch("client-a", "倾听", limit=5)

    assert consolidation["processed_checkpoint_ids"] == ["checkpoint-5"]
    assert consolidation["last_turn_count"] == 5
    assert consolidation["last_message_id"] == "ai_req-5"
    assert [memory.content for memory in memories] == [
        "用户希望难受时先被倾听，不要被急着建议。"
    ]
    assert first == {"memory_warning": ""}
    assert second == {"memory_warning": ""}


@pytest.mark.asyncio
async def test_consolidation_uses_deterministic_request_fallback_checkpoint(
    deps, runtime, writer, monkeypatch
):
    """Catches pre-checkpoint execution losing its idempotency key."""
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "true")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_INTERVAL", "1")
    state = {
        "request_id": "req-1",
        "turn_count": 1,
        "messages": [HumanMessage(id="human_req-1", content="今天有点累。")],
        "thread_meta": {"thread_id": "thread-a"},
    }

    update = await maybe_consolidate(
        state,
        runtime,
        writer,
        {"configurable": {"thread_id": "thread-a"}},
        deps=deps,
    )
    consolidation = await deps.memory_repository.aget_consolidation_state("client-a")

    assert update == {"memory_warning": ""}
    assert consolidation["processed_checkpoint_ids"] == ["request:req-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node", "expected_warning", "log_message"),
    [
        (extract_memory, "memory_write_failed", "memory extraction write failed"),
        (
            maybe_consolidate,
            "memory_consolidation_failed",
            "memory consolidation failed",
        ),
    ],
)
async def test_memory_store_failures_are_nonfatal_and_correlated(
    deps, runtime, writer, caplog, monkeypatch, node, expected_warning, log_message
):
    """Catches Store outages escaping the graph or losing request correlation."""
    monkeypatch.setenv("MEMORY_CONSOLIDATION_INTERVAL", "1")
    failing_deps = replace(deps, memory_repository=FailingMemoryRepository())
    state = completed_turn_state("我希望以后都用中文回答。", "好的。")

    with caplog.at_level(logging.WARNING):
        update = await node(
            state,
            runtime,
            writer,
            {"configurable": {"thread_id": "thread-a"}},
            deps=failing_deps,
        )

    assert update == {"memory_warning": expected_warning}
    assert log_message in caplog.text
    assert "request_id=req-1" in caplog.text
    assert "thread_id=thread-a" in caplog.text
    assert writer.events == []
