from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.nodes.generation import CRISIS_FALLBACK_ZH_CN
from chatbot.graphs.turn import build_turn_graph
from chatbot.memory import StoreMemoryRepository
from chatbot.models.graph import GraphContext


FIXED_NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
EMOTION_JSON = (
    '{"primary_emotion":"content","confidence":0.6,'
    '"secondary_emotions":[],"evidence":"steady",'
    '"reply_strategy":"respond naturally","trajectory_note":"stable",'
    '"safety_level":"normal"}'
)


class SequenceModel:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[Any, Any]] = []

    async def ainvoke(self, value, config=None, **kwargs):
        self.calls.append((value, config))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)


class FailingMemoryRepository:
    async def asearch(self, *args, **kwargs):
        raise RuntimeError("memory unavailable")

    async def aremember(self, *args, **kwargs):
        raise RuntimeError("memory unavailable")

    async def aget_consolidation_state(self, *args, **kwargs):
        raise RuntimeError("memory unavailable")

    async def amark_consolidated(self, *args, **kwargs):
        raise RuntimeError("memory unavailable")


class CountingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def aget(self, *args, **kwargs):
        self.calls.append("get")
        return await super().aget(*args, **kwargs)

    async def aput(self, *args, **kwargs):
        self.calls.append("put")
        return await super().aput(*args, **kwargs)

    async def asearch(self, *args, **kwargs):
        self.calls.append("search")
        return await super().asearch(*args, **kwargs)


def make_deps(
    store: InMemoryStore,
    *,
    chat_model: Any | None = None,
    emotion_model: Any | None = None,
    emotion_interval: int = 5,
    memory_repository: Any | None = None,
) -> NodeDependencies:
    llm = LlmConfig(provider="test", api_key="test", model="test", temperature=0.0)
    return NodeDependencies(
        chat_model=chat_model or SequenceModel("我在听。"),
        emotion_model=emotion_model or SequenceModel(EMOTION_JSON),
        chat_config=ChatConfig(
            chat_llm=llm,
            emotion_llm=llm,
            emotion_interval=emotion_interval,
        ),
        graph_config=GraphConfig(
            checkpoint_db_path=":memory:",
            store_db_path=":memory:",
            timeline_limit=50,
            request_history_limit=64,
            strict_msgpack=True,
            client_id_signing_secret="a" * 32,
        ),
        memory_repository=memory_repository
        or StoreMemoryRepository(store, now=lambda: FIXED_NOW),
        now=lambda: FIXED_NOW,
    )


def context(request_id: str, *, client_id: str = "client-a") -> GraphContext:
    return GraphContext(client_id=client_id, request_id=request_id, locale="zh-CN")


def turn_input(request_id: str, message: str) -> dict[str, str]:
    return {"operation": "turn", "request_id": request_id, "input_message": message}


def compile_graph(deps: NodeDependencies, store: InMemoryStore):
    builder = build_turn_graph(deps)
    assert isinstance(builder, StateGraph)
    return builder.compile(checkpointer=InMemorySaver(), store=store)


async def stream_parts(graph, input_value, config, graph_context):
    return [
        part
        async for part in graph.astream(
            input_value,
            config,
            context=graph_context,
            stream_mode=["messages", "custom"],
            version="v2",
        )
    ]


def custom_events(parts) -> list[dict[str, Any]]:
    return [part["data"] for part in parts if part["type"] == "custom"]


@pytest.mark.asyncio
async def test_turn_graph_runs_first_turn_and_persists_messages():
    """Catches a broken main path or a factory that returns an already compiled graph."""
    store = InMemoryStore()
    deps = make_deps(store)
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-1"}}

    result = await graph.ainvoke(
        turn_input("req-1", "我有点担心"),
        config,
        context=context("req-1"),
    )

    assert [message.type for message in result["messages"]] == ["human", "ai"]
    assert result["turn_count"] == 1
    assert result["last_emotion_analysis_turn"] == 1
    assert result["processed_requests"]["req-1"]["content"] == "我在听。"


@pytest.mark.asyncio
async def test_completed_request_replays_without_model_or_store_access():
    """Catches idempotent replay accidentally traversing analysis, context, or generation."""
    store = CountingStore()
    chat_model = SequenceModel("第一条回复")
    emotion_model = SequenceModel(EMOTION_JSON)
    deps = make_deps(store, chat_model=chat_model, emotion_model=emotion_model)
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-replay"}}
    graph_context = context("req-replay")
    await graph.ainvoke(
        turn_input("req-replay", "普通消息"), config, context=graph_context
    )
    store.calls.clear()

    parts = await stream_parts(
        graph,
        turn_input("req-replay", "重复提交"),
        config,
        graph_context,
    )

    assert len(chat_model.calls) == 1
    assert len(emotion_model.calls) == 1
    assert store.calls == []
    assert custom_events(parts) == [
        {
            "event": "done",
            "data": {
                "message_id": "ai_req-replay",
                "content": "第一条回复",
                "replayed": True,
            },
        }
    ]
    snapshot = await graph.aget_state(config)
    assert [message.id for message in snapshot.values["messages"]] == [
        "human_req-replay",
        "ai_req-replay",
    ]


@pytest.mark.asyncio
async def test_interval_not_due_reuses_checkpointed_emotion():
    """Catches routine turns invoking the emotion model before the configured interval."""
    store = InMemoryStore()
    chat_model = SequenceModel("回复一", "回复二")
    emotion_model = SequenceModel(EMOTION_JSON)
    deps = make_deps(store, chat_model=chat_model, emotion_model=emotion_model)
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-interval"}}

    await graph.ainvoke(turn_input("req-1", "第一条"), config, context=context("req-1"))
    result = await graph.ainvoke(
        turn_input("req-2", "第二条"), config, context=context("req-2")
    )

    assert len(emotion_model.calls) == 1
    assert result["turn_count"] == 2
    assert result["last_emotion_analysis_turn"] == 1


@pytest.mark.asyncio
async def test_risk_signal_forces_emotion_analysis_before_interval():
    """Catches risk signals being subordinated to the ordinary analysis interval."""
    store = InMemoryStore()
    emotion_model = SequenceModel(EMOTION_JSON)
    deps = make_deps(store, emotion_model=emotion_model)
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-risk"}}

    await graph.ainvoke(turn_input("req-1", "第一条"), config, context=context("req-1"))
    result = await graph.ainvoke(
        turn_input("req-2", "我在讨论自杀预防新闻"),
        config,
        context=context("req-2"),
    )

    assert len(emotion_model.calls) == 2
    assert result["last_emotion_analysis_turn"] == 2
    assert result["messages"][-1].additional_kwargs["safety_level"] == "normal"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_level"),
    [("今天过得还不错", "normal"), ("我真的快崩溃了", "supportive")],
)
async def test_normal_and_supportive_turns_use_ordinary_generation(message, expected_level):
    """Catches non-crisis safety levels falling into the crisis-only output path."""
    store = InMemoryStore()
    deps = make_deps(store, chat_model=SequenceModel(f"{expected_level} reply"))
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": f"thread-{expected_level}"}}

    parts = await stream_parts(
        graph,
        turn_input("req-1", message),
        config,
        context("req-1"),
    )

    assert not [event for event in custom_events(parts) if event["event"] == "token"]
    snapshot = await graph.aget_state(config)
    assert snapshot.values["messages"][-1].additional_kwargs["safety_level"] == expected_level


@pytest.mark.asyncio
async def test_explicit_crisis_uses_crisis_generation_and_one_validated_token():
    """Catches explicit first-person crisis language leaking into ordinary generation."""
    store = InMemoryStore()
    reply = "请先离开危险处，并马上联系你信任的人。"
    deps = make_deps(store, chat_model=SequenceModel(reply))
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-crisis"}}

    parts = await stream_parts(
        graph,
        turn_input("req-1", "我现在想自杀"),
        config,
        context("req-1"),
    )

    token_events = [event for event in custom_events(parts) if event["event"] == "token"]
    assert token_events == [{"event": "token", "data": {"content": reply}}]
    snapshot = await graph.aget_state(config)
    assert snapshot.values["messages"][-1].additional_kwargs["safety_level"] == "crisis"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_reply", "expected_token"),
    [
        ("危机完整回复", "危机完整回复"),
        ("   ", CRISIS_FALLBACK_ZH_CN),
    ],
    ids=["validated-reply", "validated-fallback"],
)
async def test_crisis_stream_withholds_runnable_chunks_until_validation(
    model_reply, expected_token
):
    """Catches callback-bearing crisis models leaking chunks before validation."""
    store = InMemoryStore()
    deps = make_deps(
        store,
        chat_model=FakeListChatModel(responses=[model_reply]),
        emotion_model=SequenceModel(EMOTION_JSON),
    )
    graph = compile_graph(deps, store)

    parts = await stream_parts(
        graph,
        turn_input("req-crisis-stream", "我现在想自杀"),
        {"configurable": {"thread_id": "thread-crisis-stream"}},
        context("req-crisis-stream"),
    )

    crisis_chunks = [
        message
        for part in parts
        if part["type"] == "messages"
        for message, metadata in [part["data"]]
        if isinstance(message, AIMessageChunk)
        and metadata["langgraph_node"] == "generate_crisis_reply"
    ]
    token_events = [
        event for event in custom_events(parts) if event["event"] == "token"
    ]
    assert crisis_chunks == []
    assert token_events == [
        {"event": "token", "data": {"content": expected_token}}
    ]


@pytest.mark.asyncio
async def test_emotion_failure_cannot_downgrade_explicit_crisis():
    """Catches an unavailable emotion provider bypassing deterministic crisis handling."""
    store = InMemoryStore()
    reply = "先确保安全，并联系可信任的人。"
    deps = make_deps(
        store,
        chat_model=SequenceModel(reply),
        emotion_model=SequenceModel(RuntimeError("emotion unavailable")),
    )
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-crisis-failure"}}

    parts = await stream_parts(
        graph,
        turn_input("req-1", "我现在想自杀"),
        config,
        context("req-1"),
    )

    names = [event["event"] for event in custom_events(parts)]
    assert names == ["user_message", "emotion_start", "emotion_error", "safety", "token", "done"]
    snapshot = await graph.aget_state(config)
    assert snapshot.values["messages"][-1].additional_kwargs["safety_level"] == "crisis"
    assert snapshot.values.get("last_emotion_analysis_turn", 0) == 0


@pytest.mark.asyncio
async def test_memory_failures_do_not_prevent_turn_completion(caplog):
    """Catches long-term-memory outages escaping the reply and finalization path."""
    store = InMemoryStore()
    deps = make_deps(
        store,
        chat_model=SequenceModel("仍然完成回复"),
        memory_repository=FailingMemoryRepository(),
    )
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-memory-failure"}}

    parts = await stream_parts(
        graph,
        turn_input("req-1", "我希望以后都用中文回答"),
        config,
        context("req-1"),
    )

    assert custom_events(parts)[-1]["event"] == "done"
    snapshot = await graph.aget_state(config)
    assert snapshot.values["processed_requests"]["req-1"]["content"] == "仍然完成回复"
    assert snapshot.values["memory_warning"] == ""
    assert "memory context read failed" in caplog.text
    assert "memory extraction write failed" in caplog.text


@pytest.mark.asyncio
async def test_compiled_graph_reuses_real_repository_state_across_threads():
    """Catches graph construction replacing the shared repository between invocations."""
    store = InMemoryStore()
    chat_model = SequenceModel("已记住", "会继续使用中文")
    repository = StoreMemoryRepository(store, now=lambda: FIXED_NOW)
    deps = make_deps(store, chat_model=chat_model, memory_repository=repository)
    graph = compile_graph(deps, store)

    await graph.ainvoke(
        turn_input("req-store", "我希望以后都用中文回答"),
        {"configurable": {"thread_id": "thread-store"}},
        context=context("req-store"),
    )
    await graph.ainvoke(
        turn_input("req-read", "请继续用中文回答"),
        {"configurable": {"thread_id": "thread-read"}},
        context=context("req-read"),
    )

    second_prompt = chat_model.calls[1][0].to_messages()[0].content
    assert "Relevant Long-term Memory:" in second_prompt
    assert "用户希望以后都用中文回答。" in second_prompt
    stored = await store.asearch(("client-a", "memories"))
    assert len(stored) == 1
    assert stored[0].value["use_count"] == 1


@pytest.mark.asyncio
async def test_normal_generation_failure_has_clean_checkpoint_and_safe_retry():
    """Catches failed generation leaving partial assistant state or duplicating user input."""
    store = InMemoryStore()
    chat_model = SequenceModel(RuntimeError("chat unavailable"), "重试成功")
    deps = make_deps(store, chat_model=chat_model)
    graph = compile_graph(deps, store)
    config = {"configurable": {"thread_id": "thread-retry"}}
    input_value = turn_input("req-retry", "请重试这条消息")
    graph_context = context("req-retry")

    with pytest.raises(RuntimeError, match="chat unavailable"):
        await graph.ainvoke(input_value, config, context=graph_context)

    failed = await graph.aget_state(config)
    assert [message.type for message in failed.values["messages"]] == ["human"]
    assert not failed.values.get("response_content")
    assert not failed.values.get("response_message_id")

    result = await graph.ainvoke(input_value, config, context=graph_context)

    assert [message.type for message in result["messages"]] == ["human", "ai"]
    assert [message.id for message in result["messages"]].count("human_req-retry") == 1
    assert result["turn_count"] == 1
    assert result["processed_requests"]["req-retry"]["content"] == "重试成功"


@pytest.mark.asyncio
async def test_v2_stream_distinguishes_internal_and_visible_model_messages():
    """Catches internal emotion chunks being indistinguishable from user-visible generation."""
    store = InMemoryStore()
    emotion_model = FakeListChatModel(responses=[EMOTION_JSON])
    chat_model = FakeListChatModel(responses=["可见回复"])
    deps = make_deps(store, chat_model=chat_model, emotion_model=emotion_model)
    graph = compile_graph(deps, store)

    parts = await stream_parts(
        graph,
        turn_input("req-stream", "普通消息"),
        {"configurable": {"thread_id": "thread-stream"}},
        context("req-stream"),
    )

    events = custom_events(parts)
    names = [event["event"] for event in events]
    assert "emotion_start" in names
    assert "emotion_done" in names
    assert "safety" in names
    assert "done" in names
    model_chunks = [
        part["data"]
        for part in parts
        if part["type"] == "messages"
        and isinstance(part["data"][0], AIMessageChunk)
    ]
    assert {metadata["langgraph_node"] for _, metadata in model_chunks} == {
        "analyze_emotion",
        "generate_reply",
    }
    visible_message_parts = [
        item for item in model_chunks if item[1]["langgraph_node"] == "generate_reply"
    ]
    assert visible_message_parts
    assert all(
        metadata["langgraph_node"] == "generate_reply"
        for _, metadata in visible_message_parts
    )
    assert "".join(str(message.content) for message, _ in visible_message_parts) == "可见回复"
