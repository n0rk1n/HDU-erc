from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.onboarding import build_onboarding_graph
from chatbot.models.graph import GraphContext


class StructuredDraftModel:
    def __init__(self, result: Any = None, *, error: Exception | None = None):
        self.result = result
        self.error = error
        self.schemas: list[Any] = []
        self.calls: list[tuple[Any, Any]] = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    async def ainvoke(self, value, config=None, **kwargs):
        self.calls.append((value, config))
        if self.error is not None:
            raise self.error
        return self.result


class CountingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.put_count = 0

    async def aput(self, *args, **kwargs):
        self.put_count += 1
        return await super().aput(*args, **kwargs)


def make_deps(model: StructuredDraftModel) -> NodeDependencies:
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
        memory_repository=object(),
        now=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def context() -> GraphContext:
    return GraphContext(client_id="client-a", request_id="profile-1", locale="zh-CN")


def profile_input() -> dict[str, Any]:
    return {
        "operation": "onboard",
        "request_id": "profile-1",
        "profile_answers": [
            {"key": "response_style", "answer": " 简短 "},
            {"key": "unknown", "answer": "不应进入模型"},
            {"key": "avoidance", "answer": "   "},
            {"key": "life_stage", "answer": 123},
        ],
        "target_message_id": "stale-target",
        "regeneration_reason": "stale-reason",
        "response_content": "stale-response",
        "error_code": "stale-error",
    }


def compile_graph(model: StructuredDraftModel, store: InMemoryStore):
    builder = build_onboarding_graph(make_deps(model))
    assert isinstance(builder, StateGraph)
    return builder.compile(store=store)


@pytest.mark.asyncio
async def test_onboarding_returns_sanitized_structured_model_draft():
    """Catches model drafts bypassing schema-based generation or profile sanitation."""
    store = CountingStore()
    model = StructuredDraftModel(
        {
            "response_style": " 温柔、简短 ",
            "preferred_name": "   ",
            "unknown": "不应保留",
        }
    )
    graph = compile_graph(model, store)

    result = await graph.ainvoke(
        profile_input(),
        {"tags": ["profile-draft"]},
        context=context(),
    )

    assert result["profile_draft"] == {"response_style": "温柔、简短"}
    assert len(model.schemas) == 1
    assert len(model.calls) == 1
    prompt, config = model.calls[0]
    assert "不应进入模型" not in prompt
    assert '{"key": "response_style", "answer": "简短"}' in prompt
    assert config["tags"] == ["profile-draft"]


@pytest.mark.asyncio
async def test_onboarding_uses_sanitized_fallback_when_model_fails():
    """Catches provider failure preventing the deterministic local draft."""
    store = CountingStore()
    graph = compile_graph(
        StructuredDraftModel(error=RuntimeError("provider unavailable")), store
    )

    result = await graph.ainvoke(profile_input(), context=context())

    assert result["profile_draft"] == {"response_style": "简短"}


@pytest.mark.asyncio
async def test_onboarding_returns_draft_without_writing_profile_store():
    """Catches onboarding bypassing confirmation and persisting the draft itself."""
    store = CountingStore()
    model = StructuredDraftModel({"response_style": "简短"})
    graph = compile_graph(model, store)

    result = await graph.ainvoke(profile_input(), context=context())

    assert result["profile_draft"] == {"response_style": "简短"}
    assert store.put_count == 0
    assert await store.aget(("client-a", "profile"), "current") is None


@pytest.mark.asyncio
async def test_onboarding_draft_survives_compiled_graph_and_clears_other_transients():
    """Catches terminal cleanup dropping the caller-visible draft or leaking stale inputs."""
    store = CountingStore()
    graph = compile_graph(StructuredDraftModel({"response_style": "简短"}), store)

    result = await graph.ainvoke(profile_input(), context=context())

    assert result["profile_draft"] == {"response_style": "简短"}
    assert result["operation"] == ""
    assert result["request_id"] == ""
    assert result["profile_answers"] == []
    assert result["target_message_id"] == ""
    assert result["regeneration_reason"] == ""
    assert result["response_content"] == ""
    assert result["error_code"] == ""
