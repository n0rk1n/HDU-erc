from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.graphs.conversation import build_conversation_graph
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.nodes.input import GraphInputError
from chatbot.memory import MemoryRuntimeConfig, StoreMemoryRepository
from chatbot.models.graph import GraphContext


FIXED_NOW = datetime(2026, 9, 1, 12, 30, tzinfo=timezone.utc)
EMOTION_JSON = (
    '{"primary_emotion":"content","confidence":0.8,'
    '"secondary_emotions":[],"evidence":"steady",'
    '"reply_strategy":"respond naturally","trajectory_note":"stable",'
    '"safety_level":"normal"}'
)


class ConversationModel:
    def __init__(self, response: str) -> None:
        self.response = response

    async def ainvoke(self, value, config=None, **kwargs):
        return AIMessage(
            content=self.response,
            response_metadata={"model_name": "fresh-model"},
            usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )

    def with_structured_output(self, schema):
        return self


def graph_context(request_id: str) -> GraphContext:
    return GraphContext(client_id="client-a", request_id=request_id, locale="zh-CN")


def make_deps(store: InMemoryStore) -> NodeDependencies:
    llm = LlmConfig(provider="test", api_key="test", model="test", temperature=0.0)
    return NodeDependencies(
        chat_model=ConversationModel("新回复"),
        emotion_model=ConversationModel(EMOTION_JSON),
        chat_config=ChatConfig(chat_llm=llm, emotion_llm=llm, emotion_interval=5),
        graph_config=GraphConfig(
            checkpoint_db_path=":memory:",
            store_db_path=":memory:",
            timeline_limit=50,
            request_history_limit=64,
            strict_msgpack=True,
            client_id_signing_secret="a" * 32,
        ),
        memory_repository=StoreMemoryRepository(store, now=lambda: FIXED_NOW),
        memory_config=MemoryRuntimeConfig(enabled=True, max_results=5),
        now=lambda: FIXED_NOW,
    )


def compiled_graph(store: InMemoryStore):
    return build_conversation_graph(make_deps(store)).compile(
        checkpointer=InMemorySaver(),
        store=store,
    )


@pytest.mark.asyncio
async def test_conversation_graph_routes_turn_to_observable_reply():
    """Catches the parent routing a turn to the wrong child graph."""
    store = InMemoryStore()
    graph = compiled_graph(store)

    result = await graph.ainvoke(
        {"operation": "turn", "request_id": "turn-1", "input_message": "你好"},
        {"configurable": {"thread_id": "thread-turn"}},
        context=graph_context("turn-1"),
    )

    assert [message.content for message in result["messages"]] == ["你好", "新回复"]
    assert result["processed_requests"]["turn-1"]["content"] == "新回复"


@pytest.mark.asyncio
async def test_conversation_graph_routes_regeneration_to_same_id_replacement():
    """Catches the parent routing regeneration to turn or onboarding behavior."""
    store = InMemoryStore()
    graph = compiled_graph(store)

    result = await graph.ainvoke(
        {
            "operation": "regenerate",
            "request_id": "regen-1",
            "target_message_id": "ai-target",
            "regeneration_reason": "不准确",
            "messages": [
                HumanMessage(id="human-target", content="原问题"),
                AIMessage(id="ai-target", content="旧回复", additional_kwargs={}),
            ],
        },
        {"configurable": {"thread_id": "thread-regen"}},
        context=graph_context("regen-1"),
    )

    assert [message.id for message in result["messages"]] == ["human-target", "ai-target"]
    assert result["messages"][-1].content == "新回复"
    assert result["messages"][-1].additional_kwargs["original_content"] == "旧回复"


@pytest.mark.asyncio
async def test_conversation_graph_routes_onboarding_to_profile_draft():
    """Catches the parent routing onboarding through a message-generating child."""
    store = InMemoryStore()
    graph = compiled_graph(store)

    result = await graph.ainvoke(
        {
            "operation": "onboard",
            "request_id": "profile-1",
            "profile_answers": [{"key": "response_style", "answer": "简短"}],
        },
        {"configurable": {"thread_id": "thread-profile"}},
        context=graph_context("profile-1"),
    )

    assert result["profile_draft"] == {"response_style": "简短"}
    assert result.get("messages", []) == []


@pytest.mark.asyncio
async def test_conversation_graph_rejects_unknown_operation_with_stable_error():
    """Catches unknown operations falling through to an arbitrary child graph."""
    store = InMemoryStore()

    with pytest.raises(GraphInputError, match="invalid_operation") as exc_info:
        await compiled_graph(store).ainvoke(
            {"operation": "unknown"},
            {"configurable": {"thread_id": "thread-invalid"}},
            context=graph_context("invalid-1"),
        )
    assert exc_info.value.args[0] == "invalid_operation"
