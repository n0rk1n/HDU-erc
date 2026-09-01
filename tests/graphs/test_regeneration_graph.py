from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.regeneration import RegenerationError, build_regeneration_graph
from chatbot.models.graph import GraphContext


FIXED_NOW = datetime(2026, 9, 1, 12, 30, tzinfo=timezone.utc)
REASONS = (
    "不准确",
    "不完整",
    "没有理解我的问题",
    "语气不合适",
    "其他",
)


class RecordingModel:
    def __init__(self, response: str = "新回复", *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    async def ainvoke(self, value, config=None, **kwargs):
        self.calls.append((value, config))
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.response)


class RecordingMemoryRepository:
    def __init__(self) -> None:
        self.searches: list[tuple[str, str, int]] = []

    async def asearch(self, client_id: str, query: str, *, limit: int):
        self.searches.append((client_id, query, limit))
        return []


def make_deps(model: RecordingModel, memory_repository) -> NodeDependencies:
    llm = LlmConfig(provider="test", api_key="test", model="test", temperature=0.0)
    return NodeDependencies(
        chat_model=model,
        emotion_model=object(),
        chat_config=ChatConfig(chat_llm=llm, emotion_llm=llm, emotion_interval=5),
        graph_config=GraphConfig(
            checkpoint_db_path=":memory:",
            store_db_path=":memory:",
            timeline_limit=50,
            request_history_limit=64,
            strict_msgpack=True,
            client_id_signing_secret="a" * 32,
        ),
        memory_repository=memory_repository,
        now=lambda: FIXED_NOW,
    )


def graph_context() -> GraphContext:
    return GraphContext(client_id="client-a", request_id="regen-1", locale="zh-CN")


def regeneration_input(*, reason: str = "不准确") -> dict[str, Any]:
    return {
        "operation": "regenerate",
        "request_id": "regen-1",
        "target_message_id": "ai-target",
        "regeneration_reason": reason,
        "messages": [
            HumanMessage(id="human-1", content="早先问题"),
            AIMessage(id="ai-1", content="早先回复"),
            HumanMessage(id="human-target", content="原问题"),
            AIMessage(
                id="ai-target",
                content="旧回复",
                additional_kwargs={
                    "feedback": "dislike",
                    "turn_count": 2,
                    "emotion_state": {
                        "primary_emotion": "anxious",
                        "confidence": 0.8,
                    },
                    "predicted_emotion": "anxious",
                    "safety_level": "supportive",
                    "safety_note": "keep-supportive",
                },
            ),
            HumanMessage(id="human-later", content="之后的问题"),
            AIMessage(id="ai-later", content="之后的回复"),
        ],
        "turn_count": 3,
        "emotion_state": {"primary_emotion": "content", "confidence": 0.9},
        "emotion_timeline": [
            {"turn_count": 2, "primary_emotion": "anxious"},
            {"turn_count": 3, "primary_emotion": "content"},
        ],
        "recent_emotions": ["anxious", "content"],
        "safety_state": {"level": "normal", "guidance": "reply naturally"},
    }


def tool_chain_input(*, target_message_id: str) -> dict[str, Any]:
    return {
        "operation": "regenerate",
        "request_id": "regen-1",
        "target_message_id": target_message_id,
        "regeneration_reason": "不准确",
        "messages": [
            HumanMessage(id="human-tool", content="请查询天气"),
            AIMessage(
                id="ai-tool-call",
                content="",
                tool_calls=[
                    {
                        "name": "lookup_weather",
                        "args": {"city": "杭州"},
                        "id": "call-weather",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                id="tool-weather",
                content="晴，26℃",
                tool_call_id="call-weather",
            ),
            AIMessage(id="ai-final", content="杭州今天晴，26℃。"),
        ],
        "turn_count": 1,
        "emotion_state": {"primary_emotion": "content", "confidence": 0.8},
        "emotion_timeline": [{"turn_count": 1, "primary_emotion": "content"}],
        "recent_emotions": ["content"],
        "safety_state": {"level": "normal", "guidance": "reply naturally"},
    }


async def compile_graph(deps: NodeDependencies, store: InMemoryStore):
    builder = build_regeneration_graph(deps)
    assert isinstance(builder, StateGraph)
    return builder.compile(store=store)


@pytest.mark.asyncio
async def test_regeneration_replaces_same_message_id_and_preserves_original_metadata():
    """Catches regeneration appending a new ID or discarding target audit metadata."""
    store = InMemoryStore()
    await store.aput(("client-a", "profile"), "current", {"response_style": "简短"})
    model = RecordingModel()
    memory_repository = RecordingMemoryRepository()
    graph = await compile_graph(make_deps(model, memory_repository), store)
    original_state = regeneration_input()

    result = await graph.ainvoke(original_state, context=graph_context())

    assert [message.id for message in result["messages"]] == [
        "human-1",
        "ai-1",
        "human-target",
        "ai-target",
        "human-later",
        "ai-later",
    ]
    replacement = next(
        message for message in result["messages"] if message.id == "ai-target"
    )
    assert replacement.content == "新回复"
    assert replacement.additional_kwargs == {
        "feedback": None,
        "turn_count": 2,
        "emotion_state": {
            "primary_emotion": "anxious",
            "confidence": 0.8,
        },
        "predicted_emotion": "anxious",
        "safety_level": "supportive",
        "safety_note": "keep-supportive",
        "original_content": "旧回复",
        "regeneration_reason": "不准确",
        "regenerated_at": "2026-09-01T12:30:00+00:00",
        "regenerated": True,
    }
    assert result["processed_requests"]["regen-1"]["status"] == "completed"
    assert result["processed_requests"]["regen-1"]["response_message_id"] == "ai-target"


@pytest.mark.asyncio
async def test_regeneration_rejects_intermediate_tool_call_and_preserves_transcript():
    """Catches same-ID replacement deleting a tool call required by later ToolMessage."""
    store = InMemoryStore()
    builder = build_regeneration_graph(
        make_deps(RecordingModel(), RecordingMemoryRepository())
    )
    graph = builder.compile(checkpointer=InMemorySaver(), store=store)
    state = tool_chain_input(target_message_id="ai-tool-call")
    original_messages = list(state["messages"])
    config = {"configurable": {"thread_id": "thread-tool-reject"}}

    with pytest.raises(RegenerationError) as exc_info:
        await graph.ainvoke(state, config, context=graph_context())

    snapshot = await graph.aget_state(config)
    assert exc_info.value.code == "non_ai_target"
    assert snapshot.values["messages"] == original_messages
    assert snapshot.values["messages"][1].tool_calls == [
        {
            "name": "lookup_weather",
            "args": {"city": "杭州"},
            "id": "call-weather",
            "type": "tool_call",
        }
    ]
    assert snapshot.values["messages"][2].tool_call_id == "call-weather"


@pytest.mark.asyncio
async def test_regeneration_accepts_final_tool_chain_reply_and_replaces_only_final_id():
    """Catches final reply validation damaging the preceding valid tool-call chain."""
    store = InMemoryStore()
    graph = await compile_graph(
        make_deps(RecordingModel("新的最终回复"), RecordingMemoryRepository()), store
    )
    state = tool_chain_input(target_message_id="ai-final")

    result = await graph.ainvoke(state, context=graph_context())

    assert [message.id for message in result["messages"]] == [
        "human-tool",
        "ai-tool-call",
        "tool-weather",
        "ai-final",
    ]
    assert result["messages"][1].tool_calls == [
        {
            "name": "lookup_weather",
            "args": {"city": "杭州"},
            "id": "call-weather",
            "type": "tool_call",
        }
    ]
    assert result["messages"][2].tool_call_id == "call-weather"
    assert result["messages"][3].content == "新的最终回复"
    assert result["messages"][3].additional_kwargs["original_content"] == (
        "杭州今天晴，26℃。"
    )


@pytest.mark.asyncio
async def test_regeneration_rejects_ai_with_invalid_tool_calls():
    """Catches malformed intermediate tool requests being treated as visible replies."""
    store = InMemoryStore()
    graph = await compile_graph(
        make_deps(RecordingModel(), RecordingMemoryRepository()), store
    )
    state = regeneration_input()
    target = next(message for message in state["messages"] if message.id == "ai-target")
    target.invalid_tool_calls = [
        {
            "name": "lookup_weather",
            "args": "{",
            "id": "call-invalid",
            "error": "invalid json",
            "type": "invalid_tool_call",
        }
    ]

    with pytest.raises(RegenerationError) as exc_info:
        await graph.ainvoke(state, context=graph_context())

    assert exc_info.value.code == "non_ai_target"


@pytest.mark.asyncio
async def test_regeneration_rebuilds_only_through_target_and_queries_original_question():
    """Catches regeneration using later dialogue or a stale input for retrieval/generation."""
    store = InMemoryStore()
    await store.aput(("client-a", "profile"), "current", {"response_style": "简短"})
    model = RecordingModel()
    memory_repository = RecordingMemoryRepository()
    graph = await compile_graph(make_deps(model, memory_repository), store)

    await graph.ainvoke(
        regeneration_input(reason="语气不合适"),
        {"tags": ["regen-stream"]},
        context=graph_context(),
    )

    assert memory_repository.searches
    client_id, query, _ = memory_repository.searches[0]
    assert client_id == "client-a"
    assert "原问题" in query
    assert "之后的问题" not in query
    prompt_value, received_config = model.calls[0]
    prompt_messages = prompt_value.to_messages()
    contents = [str(message.content) for message in prompt_messages]
    assert contents[-1] == "原问题"
    assert "早先问题" in contents
    assert "早先回复" in contents
    assert "旧回复" not in contents
    assert "之后的问题" not in contents
    assert "简短" in contents[0]
    assert any("语气不合适" in content for content in contents)
    assert received_config["tags"] == ["regen-stream"]


@pytest.mark.asyncio
async def test_regeneration_does_not_change_turn_emotion_timeline_or_extract_memory():
    """Catches regeneration being treated as a new user turn with duplicate side effects."""
    store = InMemoryStore()
    model = RecordingModel()
    memory_repository = RecordingMemoryRepository()
    graph = await compile_graph(make_deps(model, memory_repository), store)
    input_state = regeneration_input()

    result = await graph.ainvoke(input_state, context=graph_context())

    assert result["turn_count"] == 3
    assert result["emotion_state"] == {
        "primary_emotion": "content",
        "confidence": 0.9,
    }
    assert result["emotion_timeline"] == [
        {"turn_count": 2, "primary_emotion": "anxious"},
        {"turn_count": 3, "primary_emotion": "content"},
    ]
    assert result["recent_emotions"] == ["anxious", "content"]
    assert memory_repository.searches and len(memory_repository.searches) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", REASONS)
async def test_regeneration_accepts_every_existing_ui_reason(reason):
    """Catches the graph contract drifting from one of the five existing UI values."""
    store = InMemoryStore()
    graph = await compile_graph(
        make_deps(RecordingModel(), RecordingMemoryRepository()), store
    )

    result = await graph.ainvoke(
        regeneration_input(reason=reason), context=graph_context()
    )

    target = next(message for message in result["messages"] if message.id == "ai-target")
    assert target.additional_kwargs["regeneration_reason"] == reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_update", "error_code"),
    [
        ({"target_message_id": "ai-missing"}, "missing_target"),
        ({"target_message_id": "human-target"}, "non_ai_target"),
        ({"regeneration_reason": "too long"}, "invalid_reason"),
    ],
)
async def test_regeneration_validation_uses_stable_domain_codes(
    input_update, error_code
):
    """Catches validation leaking implementation-specific lookup or type errors."""
    store = InMemoryStore()
    graph = await compile_graph(
        make_deps(RecordingModel(), RecordingMemoryRepository()), store
    )

    with pytest.raises(RegenerationError) as exc_info:
        await graph.ainvoke(
            {**regeneration_input(), **input_update}, context=graph_context()
        )
    assert exc_info.value.code == error_code


@pytest.mark.asyncio
async def test_regeneration_rejects_target_that_was_already_regenerated():
    """Catches the default single-regeneration limit being bypassed."""
    store = InMemoryStore()
    graph = await compile_graph(
        make_deps(RecordingModel(), RecordingMemoryRepository()), store
    )
    state = regeneration_input()
    target = next(message for message in state["messages"] if message.id == "ai-target")
    target.additional_kwargs["regenerated"] = True

    with pytest.raises(RegenerationError) as exc_info:
        await graph.ainvoke(state, context=graph_context())
    assert exc_info.value.code == "already_regenerated"


@pytest.mark.asyncio
async def test_regeneration_maps_model_errors_to_generation_failed():
    """Catches provider details escaping instead of the stable generation domain code."""
    store = InMemoryStore()
    deps = make_deps(
        RecordingModel(error=RuntimeError("provider secret detail")),
        RecordingMemoryRepository(),
    )
    graph = await compile_graph(deps, store)

    with pytest.raises(RegenerationError) as exc_info:
        await graph.ainvoke(regeneration_input(), context=graph_context())
    assert exc_info.value.code == "generation_failed"


@pytest.mark.asyncio
async def test_regeneration_emits_done_with_replaced_target_payload():
    """Catches terminal events exposing a new ID or omitting regeneration metadata."""
    store = InMemoryStore()
    graph = await compile_graph(
        make_deps(RecordingModel(), RecordingMemoryRepository()), store
    )

    events = [
        event
        async for event in graph.astream(
            regeneration_input(reason="其他"),
            context=graph_context(),
            stream_mode="custom",
        )
    ]

    assert events == [
        {
            "event": "done",
            "data": {
                "message_id": "ai-target",
                "content": "新回复",
                "reason": "其他",
                "regenerated": True,
            },
        }
    ]
