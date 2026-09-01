from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.graphs.runtime import (
    ConversationRuntime,
    RuntimeOperationError,
    build_graph_runtime,
)
from chatbot.graphs.regeneration import RegenerationError
from chatbot.persistence.runtime import open_persistence
from chatbot.persistence.threads import ThreadRepository


FIXED_NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
EMOTION_JSON = (
    '{"primary_emotion":"content","confidence":0.8,'
    '"secondary_emotions":[],"evidence":"steady",'
    '"reply_strategy":"respond naturally","trajectory_note":"stable",'
    '"safety_level":"normal"}'
)


class StaticModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def ainvoke(self, value, config=None, **kwargs):
        self.calls += 1
        return AIMessage(content=self.response)

    def with_structured_output(self, schema):
        return self


class ConcurrentChatModel:
    def __init__(self) -> None:
        self.thread_a = ""
        self.thread_b = ""
        self.entered: list[str] = []
        self.active = defaultdict(int)
        self.max_active = defaultdict(int)
        self.first_a_release = asyncio.Event()
        self.b_release = asyncio.Event()
        self.first_a_entered = asyncio.Event()
        self.second_a_entered = asyncio.Event()
        self.b_entered = asyncio.Event()

    async def ainvoke(self, value, config=None, **kwargs):
        thread_id = config["configurable"]["thread_id"]
        self.entered.append(thread_id)
        self.active[thread_id] += 1
        self.max_active[thread_id] = max(
            self.max_active[thread_id], self.active[thread_id]
        )
        try:
            if thread_id == self.thread_a and self.entered.count(self.thread_a) == 1:
                self.first_a_entered.set()
                await self.first_a_release.wait()
            elif thread_id == self.thread_a:
                self.second_a_entered.set()
            elif thread_id == self.thread_b:
                self.b_entered.set()
                await self.b_release.wait()
            return AIMessage(content="回复")
        finally:
            self.active[thread_id] -= 1


class CancellingChatModel:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def ainvoke(self, value, config=None, **kwargs):
        self.entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class GatedDeleteSaver(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self.delete_started = asyncio.Event()
        self.allow_delete = asyncio.Event()

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_started.set()
        await self.allow_delete.wait()
        await super().adelete_thread(thread_id)


class GatedUpdateGraph:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.update_calls = 0
        self.update_started = asyncio.Event()
        self.allow_update = asyncio.Event()

    async def aupdate_state(self, *args, **kwargs):
        self.update_calls += 1
        if self.update_calls == 1:
            self.update_started.set()
            await self.allow_update.wait()
        return await self.graph.aupdate_state(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.graph, name)


def configs(tmp_path=None) -> tuple[ChatConfig, GraphConfig]:
    llm = LlmConfig(provider="test", api_key="test", model="test", temperature=0.0)
    graph = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3") if tmp_path else ":memory:",
        store_db_path=str(tmp_path / "store.sqlite3") if tmp_path else ":memory:",
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )
    return ChatConfig(chat_llm=llm, emotion_llm=llm, emotion_interval=5), graph


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def memory_runtime(
    *,
    chat_model: Any | None = None,
    emotion_model: Any | None = None,
    saver: InMemorySaver | None = None,
) -> tuple[ConversationRuntime, InMemorySaver, InMemoryStore]:
    saver = saver or InMemorySaver()
    store = InMemoryStore()
    chat_config, graph_config = configs()
    runtime = build_graph_runtime(
        SimpleNamespace(checkpointer=saver, store=store),
        chat_config,
        graph_config,
        model_factory=lambda config: (
            chat_model or StaticModel("回复"),
            emotion_model or StaticModel(EMOTION_JSON),
        ),
        now=lambda: FIXED_NOW,
    )
    return runtime, saver, store


async def consume(stream) -> list[Any]:
    return [part async for part in stream]


async def open_and_consume(stream_awaitable) -> list[Any]:
    return await consume(await stream_awaitable)


@pytest.mark.asyncio
async def test_stream_open_prevalidates_ownership_before_returning_iterator():
    """Catches ownership errors being deferred until HTTP streaming has begun."""
    runtime, _, _ = memory_runtime()
    record = await runtime.acreate_thread("client-a")

    with pytest.raises(RuntimeOperationError) as turn_error:
        await runtime.astream_turn("client-b", record.thread_id, "turn-1", "你好")
    with pytest.raises(RuntimeOperationError) as regeneration_error:
        await runtime.astream_regeneration(
            "client-b", record.thread_id, "regen-1", "ai-missing", "其他"
        )

    assert turn_error.value.code == "thread_not_found"
    assert regeneration_error.value.code == "thread_not_found"


@pytest.mark.asyncio
async def test_stream_revalidates_if_thread_deleted_after_preflight():
    """Catches a pre-opened stream resurrecting a thread deleted before iteration."""
    runtime, saver, _ = memory_runtime()
    record = await runtime.acreate_thread("client-a")
    stream = await runtime.astream_turn(
        "client-a", record.thread_id, "turn-1", "不应写入"
    )

    await runtime.adelete_thread("client-a", record.thread_id)

    with pytest.raises(RuntimeOperationError) as exc_info:
        await consume(stream)
    assert exc_info.value.code == "thread_not_found"
    assert await saver.aget_tuple(thread_config(record.thread_id)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["profile", "read"])
async def test_delete_serializes_checkpoint_reads_and_profile_writes(operation):
    """Catches checkpoint reads or profile writes racing a same-thread deletion."""
    saver = GatedDeleteSaver()
    runtime, _, _ = memory_runtime(saver=saver)
    record = await runtime.acreate_thread("client-a")
    delete_task = asyncio.create_task(
        runtime.adelete_thread("client-a", record.thread_id)
    )
    await asyncio.wait_for(saver.delete_started.wait(), timeout=2)

    if operation == "profile":
        operation_task = asyncio.create_task(
            runtime.ainvoke_profile_draft(
                "client-a",
                record.thread_id,
                "profile-1",
                [{"key": "response_style", "answer": "简短"}],
            )
        )
    else:
        operation_task = asyncio.create_task(
            runtime.aget_state("client-a", record.thread_id)
        )
    await asyncio.sleep(0)

    assert operation_task.done() is False
    saver.allow_delete.set()
    await delete_task
    with pytest.raises(RuntimeOperationError) as exc_info:
        await operation_task

    assert exc_info.value.code == "thread_not_found"
    assert await saver.aget_tuple(thread_config(record.thread_id)) is None


@pytest.mark.asyncio
async def test_thread_listing_waits_for_initial_checkpoint_before_reconciliation():
    """Catches reconciliation deleting a Store record while creation initializes Saver."""
    base, saver, _ = memory_runtime()
    gated_graph = GatedUpdateGraph(base.graph)
    runtime = ConversationRuntime(
        gated_graph,
        base.thread_repository,
        saver,
        dependencies=base.dependencies,
    )
    create_task = asyncio.create_task(
        runtime.acreate_thread("client-a", title="初始化中")
    )
    await asyncio.wait_for(gated_graph.update_started.wait(), timeout=2)
    pending_records = await runtime.thread_repository.list("client-a")
    assert len(pending_records) == 1

    list_task = asyncio.create_task(runtime.alist_threads("client-a"))
    await asyncio.sleep(0)
    assert list_task.done() is False

    gated_graph.allow_update.set()
    created = await create_task
    listed = await list_task

    assert [record.thread_id for record in listed] == [created.thread_id]
    assert await runtime.thread_repository.owns("client-a", created.thread_id) is True
    assert await saver.aget_tuple(thread_config(created.thread_id)) is not None


@pytest.mark.asyncio
async def test_catalog_initialization_for_different_clients_remains_concurrent():
    """Catches one client's catalog initialization globally blocking another client."""
    base, saver, _ = memory_runtime()
    gated_graph = GatedUpdateGraph(base.graph)
    runtime = ConversationRuntime(
        gated_graph,
        base.thread_repository,
        saver,
        dependencies=base.dependencies,
    )
    first_client = asyncio.create_task(runtime.acreate_thread("client-a"))
    await asyncio.wait_for(gated_graph.update_started.wait(), timeout=2)

    second = await asyncio.wait_for(
        runtime.acreate_thread("client-b"),
        timeout=2,
    )

    assert await runtime.thread_repository.owns("client-b", second.thread_id) is True
    gated_graph.allow_update.set()
    await first_client


@pytest.mark.asyncio
async def test_build_runtime_constructs_models_once_and_compiles_parent_once():
    """Catches per-request model or graph construction in the runtime facade."""
    saver = InMemorySaver()
    store = InMemoryStore()
    chat_config, graph_config = configs()
    calls: list[ChatConfig] = []

    def model_factory(config):
        calls.append(config)
        return StaticModel("回复"), StaticModel(EMOTION_JSON)

    runtime = build_graph_runtime(
        SimpleNamespace(checkpointer=saver, store=store),
        chat_config,
        graph_config,
        model_factory=model_factory,
        now=lambda: FIXED_NOW,
    )

    record = await runtime.acreate_thread("client-a")
    first_parts = await consume(
        await runtime.astream_turn("client-a", record.thread_id, "req-1", "你好")
    )
    await consume(
        await runtime.astream_turn("client-a", record.thread_id, "req-2", "再见")
    )

    assert calls == [chat_config]
    assert first_parts
    assert {part["type"] for part in first_parts} <= {"messages", "custom"}
    assert runtime.dependencies.memory_repository is runtime.memory_repository
    assert runtime.graph is runtime.compiled_graph


@pytest.mark.asyncio
async def test_create_thread_initializes_checkpoint_and_rolls_back_on_failure():
    """Catches directory records existing without a corresponding initial checkpoint."""
    runtime, saver, _ = memory_runtime()

    record = await runtime.acreate_thread("client-a", title="初始标题")

    checkpoint = await saver.aget_tuple(thread_config(record.thread_id))
    snapshot = await runtime.aget_state("client-a", record.thread_id)
    reconciled = await runtime.alist_threads("client-a")
    assert checkpoint is not None
    assert [item.thread_id for item in reconciled] == [record.thread_id]
    assert snapshot.values["thread_meta"] == {
        "thread_id": record.thread_id,
        "title": "初始标题",
        "created_at": "2026-09-01T10:00:00+00:00",
        "updated_at": "2026-09-01T10:00:00+00:00",
    }

    class FailingGraph:
        async def aupdate_state(self, *args, **kwargs):
            raise RuntimeError("checkpoint unavailable")

    store = InMemoryStore()
    repository = ThreadRepository(store, now=lambda: FIXED_NOW)
    failing_runtime = ConversationRuntime(FailingGraph(), repository, InMemorySaver())
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        await failing_runtime.acreate_thread("client-b")
    assert await repository.list("client-b") == []


@pytest.mark.asyncio
async def test_runtime_serializes_same_thread_and_allows_other_thread():
    """Catches same-thread mutation overlap or a global lock blocking other threads."""
    model = ConcurrentChatModel()
    runtime, _, _ = memory_runtime(chat_model=model)
    record_a = await runtime.acreate_thread("client-a", title="A")
    record_b = await runtime.acreate_thread("client-a", title="B")
    model.thread_a = record_a.thread_id
    model.thread_b = record_b.thread_id

    first_a = asyncio.create_task(
        open_and_consume(
            runtime.astream_turn("client-a", record_a.thread_id, "req-a1", "A1")
        )
    )
    await asyncio.wait_for(model.first_a_entered.wait(), timeout=2)
    second_a = asyncio.create_task(
        open_and_consume(
            runtime.astream_turn("client-a", record_a.thread_id, "req-a2", "A2")
        )
    )
    first_b = asyncio.create_task(
        open_and_consume(
            runtime.astream_turn("client-a", record_b.thread_id, "req-b1", "B1")
        )
    )
    await asyncio.wait_for(model.b_entered.wait(), timeout=2)
    await asyncio.sleep(0)

    assert runtime.lock_for(record_a.thread_id) is runtime.lock_for(record_a.thread_id)
    assert runtime.lock_for(record_a.thread_id) is not runtime.lock_for(record_b.thread_id)
    assert model.entered.count(record_a.thread_id) == 1
    assert model.entered.count(record_b.thread_id) == 1

    model.first_a_release.set()
    await asyncio.wait_for(model.second_a_entered.wait(), timeout=2)
    model.b_release.set()
    await asyncio.gather(first_a, second_a, first_b)
    assert model.max_active[record_a.thread_id] == 1


@pytest.mark.asyncio
async def test_cancelled_stream_releases_thread_lock():
    """Catches a client disconnect permanently wedging the per-thread lock."""
    model = CancellingChatModel()
    runtime, _, _ = memory_runtime(chat_model=model)
    record = await runtime.acreate_thread("client-a")
    consumer = asyncio.create_task(
        open_and_consume(
            runtime.astream_turn("client-a", record.thread_id, "req-1", "你好")
        )
    )
    await asyncio.wait_for(model.entered.wait(), timeout=2)
    assert runtime.lock_for(record.thread_id).locked() is True

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    await asyncio.wait_for(model.cancelled.wait(), timeout=2)
    assert runtime.lock_for(record.thread_id).locked() is False


@pytest.mark.asyncio
async def test_runtime_rejects_unowned_thread_for_read_stream_and_profile_draft():
    """Catches any facade path bypassing client/thread ownership validation."""
    runtime, _, _ = memory_runtime()
    record = await runtime.acreate_thread("client-a")

    with pytest.raises(RuntimeOperationError) as read_error:
        await runtime.aget_state("client-b", record.thread_id)
    with pytest.raises(RuntimeOperationError) as turn_error:
        await runtime.astream_turn("client-b", record.thread_id, "req-1", "你好")
    with pytest.raises(RuntimeOperationError) as regeneration_error:
        await runtime.astream_regeneration(
            "client-b", record.thread_id, "regen-1", "ai-missing", "其他"
        )
    with pytest.raises(RuntimeOperationError) as profile_error:
        await runtime.ainvoke_profile_draft(
            "client-b",
            record.thread_id,
            "profile-1",
            [{"key": "response_style", "answer": "简短"}],
        )
    with pytest.raises(RuntimeOperationError) as feedback_error:
        await runtime.aupdate_message_feedback(
            "client-b", record.thread_id, "ai-missing", "like"
        )
    with pytest.raises(RuntimeOperationError) as delete_error:
        await runtime.adelete_thread("client-b", record.thread_id)

    assert {
        read_error.value.code,
        turn_error.value.code,
        regeneration_error.value.code,
        profile_error.value.code,
        feedback_error.value.code,
        delete_error.value.code,
    } == {
        "thread_not_found"
    }


@pytest.mark.asyncio
async def test_feedback_replaces_same_ai_id_and_preserves_all_metadata():
    """Catches feedback appending a message or dropping provider/audit metadata."""
    runtime, _, _ = memory_runtime()
    record = await runtime.acreate_thread("client-a")
    original = AIMessage(
        id="ai-1",
        content="回复",
        additional_kwargs={"feedback": None, "safety_level": "supportive"},
        response_metadata={"model_name": "m1", "finish_reason": "stop"},
        usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        name="assistant-name",
    )
    await runtime.graph.aupdate_state(
        thread_config(record.thread_id),
        {"messages": [HumanMessage(id="human-1", content="你好"), original]},
        as_node="turn",
    )

    updated = await runtime.aupdate_message_feedback(
        "client-a", record.thread_id, "ai-1", "like"
    )
    snapshot = await runtime.aget_state("client-a", record.thread_id)

    assert updated.id == "ai-1"
    assert updated.additional_kwargs == {
        "feedback": "like",
        "safety_level": "supportive",
    }
    assert updated.response_metadata == original.response_metadata
    assert updated.usage_metadata == original.usage_metadata
    assert updated.name == "assistant-name"
    assert [message.id for message in snapshot.values["messages"]] == ["human-1", "ai-1"]

    with pytest.raises(RuntimeOperationError) as repeated:
        await runtime.aupdate_message_feedback(
            "client-a", record.thread_id, "ai-1", "dislike"
        )
    with pytest.raises(RuntimeOperationError) as invalid:
        await runtime.aupdate_message_feedback(
            "client-a", record.thread_id, "ai-1", "neutral"
        )
    with pytest.raises(RuntimeOperationError) as missing:
        await runtime.aupdate_message_feedback(
            "client-a", record.thread_id, "missing", "like"
        )
    with pytest.raises(RuntimeOperationError) as human:
        await runtime.aupdate_message_feedback(
            "client-a", record.thread_id, "human-1", "like"
        )
    assert repeated.value.code == "already_rated"
    assert invalid.value.code == "invalid_feedback"
    assert missing.value.code == "message_not_found"
    assert human.value.code == "message_not_found"


@pytest.mark.asyncio
async def test_runtime_regeneration_replays_retry_and_preserves_conflict_code():
    """Catches the facade dropping regeneration idempotency or stable domain codes."""
    chat_model = StaticModel("回复")
    runtime, _, _ = memory_runtime(chat_model=chat_model)
    record = await runtime.acreate_thread("client-a")
    await consume(
        await runtime.astream_turn("client-a", record.thread_id, "turn-1", "你好")
    )
    snapshot = await runtime.aget_state("client-a", record.thread_id)
    target_id = snapshot.values["messages"][-1].id

    first = await consume(
        await runtime.astream_regeneration(
            "client-a", record.thread_id, "regen-1", target_id, "其他"
        )
    )
    calls_after_first = chat_model.calls
    replay = await consume(
        await runtime.astream_regeneration(
            "client-a", record.thread_id, "regen-1", target_id, "其他"
        )
    )
    with pytest.raises(RegenerationError) as exc_info:
        await consume(
            await runtime.astream_regeneration(
                "client-a", record.thread_id, "regen-2", target_id, "其他"
            )
        )

    first_done = [part for part in first if part["type"] == "custom"][-1]
    replay_done = [part for part in replay if part["type"] == "custom"][-1]
    assert replay_done["data"] == first_done["data"]
    assert chat_model.calls == calls_after_first
    assert exc_info.value.code == "already_regenerated"


@pytest.mark.asyncio
async def test_delete_calls_saver_first_and_preserves_record_on_saver_failure():
    """Catches deletion orphaning Saver data or hiding a failed Saver deletion."""
    order: list[str] = []

    class TrackingSaver(InMemorySaver):
        def __init__(self, *, fail=False):
            super().__init__()
            self.fail = fail

        async def adelete_thread(self, thread_id: str) -> None:
            order.append("saver")
            if self.fail:
                raise RuntimeError("saver unavailable")
            await super().adelete_thread(thread_id)

    class TrackingRepository(ThreadRepository):
        async def delete_record(self, client_id: str, thread_id: str) -> None:
            order.append("store")
            await super().delete_record(client_id, thread_id)

    chat_config, graph_config = configs()
    store = InMemoryStore()
    saver = TrackingSaver()
    base = build_graph_runtime(
        SimpleNamespace(checkpointer=saver, store=store),
        chat_config,
        graph_config,
        model_factory=lambda config: (StaticModel("回复"), StaticModel(EMOTION_JSON)),
        now=lambda: FIXED_NOW,
    )
    repository = TrackingRepository(store, now=lambda: FIXED_NOW)
    runtime = ConversationRuntime(base.graph, repository, saver, dependencies=base.dependencies)
    record = await runtime.acreate_thread("client-a")

    await runtime.adelete_thread("client-a", record.thread_id)
    assert order == ["saver", "store"]
    assert await repository.owns("client-a", record.thread_id) is False

    order.clear()
    failing_saver = TrackingSaver(fail=True)
    failing_base = build_graph_runtime(
        SimpleNamespace(checkpointer=failing_saver, store=store),
        chat_config,
        graph_config,
        model_factory=lambda config: (StaticModel("回复"), StaticModel(EMOTION_JSON)),
        now=lambda: FIXED_NOW,
    )
    failing_runtime = ConversationRuntime(
        failing_base.graph,
        repository,
        failing_saver,
        dependencies=failing_base.dependencies,
    )
    failed = await failing_runtime.acreate_thread("client-a")

    with pytest.raises(RuntimeOperationError) as exc_info:
        await failing_runtime.adelete_thread("client-a", failed.thread_id)

    assert exc_info.value.code == "thread_delete_failed"
    assert order == ["saver"]
    assert await repository.owns("client-a", failed.thread_id) is True


@pytest.mark.asyncio
async def test_runtime_restores_thread_after_persistence_reopen(tmp_path):
    """Catches runtime state being tied to process memory instead of SQLite Checkpoint."""
    chat_config, graph_config = configs(tmp_path)
    thread_id = ""

    async with open_persistence(graph_config) as handles:
        runtime = build_graph_runtime(
            handles,
            chat_config,
            graph_config,
            model_factory=lambda config: (StaticModel("回复"), StaticModel(EMOTION_JSON)),
            now=lambda: FIXED_NOW,
        )
        record = await runtime.acreate_thread("client-a")
        thread_id = record.thread_id
        await consume(
            await runtime.astream_turn("client-a", thread_id, "req-1", "你好")
        )

    async with open_persistence(graph_config) as handles:
        reopened = build_graph_runtime(
            handles,
            chat_config,
            graph_config,
            model_factory=lambda config: (StaticModel("不应调用"), StaticModel(EMOTION_JSON)),
            now=lambda: FIXED_NOW,
        )
        snapshot = await reopened.aget_state("client-a", thread_id)

    assert [message.content for message in snapshot.values["messages"]] == ["你好", "回复"]
