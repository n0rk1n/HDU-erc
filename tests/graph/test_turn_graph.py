from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
import pytest
import pytest_asyncio
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chatbot.core.errors import InvalidMessageState
from chatbot.db.messages import MessageRepository
from chatbot.graph import (
    NodeDependencies,
    TurnContext,
    TurnState,
    compile_turn_graph,
)
from chatbot.llm.types import ModelDelta, TokenUsage
from chatbot.services.identity import IdentityService


STATE_FIELDS = {
    "user_id",
    "conversation_id",
    "request_id",
    "user_message_id",
    "assistant_message_id",
    "phase",
    "error_code",
}


class ManualClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 9, 3, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class RecordingPublisher:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.fail = fail

    async def publish(self, name: str, data: dict[str, object]) -> None:
        if self.fail:
            raise RuntimeError("publisher-secret")
        self.events.append((name, data))


class DeterministicModel:
    provider = "test-provider"

    def __init__(
        self,
        deltas: Sequence[ModelDelta],
        *,
        error: Exception | None = None,
        clock: ManualClock | None = None,
        advances: Sequence[float] = (),
    ) -> None:
        self.parameters = {
            "model": "test-model",
            "temperature": 0.2,
            "api_key": "sk-test-secret",
        }
        self.deltas = list(deltas)
        self.error = error
        self.clock = clock
        self.advances = list(advances)
        self.calls = 0
        self.prompts: list[Sequence[BaseMessage]] = []

    async def stream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[ModelDelta]:
        self.calls += 1
        self.prompts.append(prompt)
        for index, delta in enumerate(self.deltas):
            if self.clock is not None and index < len(self.advances):
                self.clock.advance(self.advances[index])
            yield delta
        if self.error is not None:
            raise self.error


class RecordingMessages:
    def __init__(self, wrapped: MessageRepository) -> None:
        self.wrapped = wrapped
        self.flushes: list[tuple[str, str | None]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    async def flush_partial(
        self,
        assistant_message_id: str,
        *,
        content: str,
        reasoning_content: str | None,
        trace: object,
    ):
        self.flushes.append((content, reasoning_content))
        return await self.wrapped.flush_partial(
            assistant_message_id,
            content=content,
            reasoning_content=reasoning_content,
            trace=trace,
        )


class FailingFlushMessages(RecordingMessages):
    async def flush_partial(
        self,
        assistant_message_id: str,
        *,
        content: str,
        reasoning_content: str | None,
        trace: object,
    ):
        raise RuntimeError("database write failed")


@pytest_asyncio.fixture
async def messages(database) -> MessageRepository:
    return MessageRepository(database)


@pytest_asyncio.fixture
async def saver(database):
    async with AsyncSqliteSaver.from_conn_string(str(database.path)) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


async def _reserved_context(
    database,
    messages: MessageRepository,
    identifier: str,
    publisher: RecordingPublisher | None = None,
) -> tuple[TurnContext, dict[str, object]]:
    conversation = (await IdentityService(database).resolve(identifier)).conversation
    request_id = str(uuid4())
    turn = await messages.reserve_turn(conversation.id, request_id, f"来自 {identifier} 的问题")
    context = TurnContext(
        thread_id=conversation.thread_id,
        user_id=conversation.user_id,
        conversation_id=conversation.id,
        request_id=request_id,
        user_message_id=turn.user.id,
        assistant_message_id=turn.assistant.id,
        publisher=publisher or RecordingPublisher(),
    )
    state: dict[str, object] = {
        "user_id": context.user_id,
        "conversation_id": context.conversation_id,
        "request_id": context.request_id,
        "user_message_id": context.user_message_id,
        "assistant_message_id": context.assistant_message_id,
        "phase": "reserved",
        "error_code": None,
    }
    return context, state


def _config(context: TurnContext) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": context.thread_id}}


def test_turn_state_contains_only_control_fields() -> None:
    """Catches checkpoint state accidentally retaining messages, clients, or secrets."""
    assert set(TurnState.__annotations__) == STATE_FIELDS


@pytest.mark.asyncio
async def test_graph_runs_real_runtime_context_nodes_in_order(
    database, messages, saver
) -> None:
    """Catches broken graph wiring or node signatures incompatible with LangGraph 1.2."""
    context, state = await _reserved_context(database, messages, "ordered")
    model = DeterministicModel(
        [
            ModelDelta(content="答", reasoning="真实依据"),
            ModelDelta(
                content="案",
                usage=TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                finish_reason="stop",
                response_metadata={"request_id": "provider-response"},
            ),
        ]
    )
    graph = compile_turn_graph(
        NodeDependencies(messages=messages, model=model), saver
    )

    result = await graph.ainvoke(state, _config(context), context=context)

    assert result["phase"] == "completed"
    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    trace = json.loads(assistant.trace_json)
    assert [node["name"] for node in trace["nodes"]] == [
        "prepare_turn",
        "generate_response",
        "finalize_turn",
    ]
    assert [node["status"] for node in trace["nodes"]] == [
        "completed",
        "completed",
        "completed",
    ]
    assert assistant.content == "答案"
    assert assistant.reasoning_content == "真实依据"
    assert context.publisher.events == [
        ("token", {"content": "答"}),
        ("token", {"content": "案"}),
    ]


@pytest.mark.asyncio
async def test_prepare_rejects_a_noncompleted_prewritten_user_message(
    database, messages, saver
) -> None:
    """Catches generation starting before the reserved user fact is durable."""
    context, state = await _reserved_context(database, messages, "invalid-reservation")
    async with database.transaction(immediate=True) as connection:
        await connection.execute(
            "UPDATE messages SET status = 'pending', completed_at = NULL WHERE id = ?",
            (context.user_message_id,),
        )
    model = DeterministicModel([ModelDelta(content="must-not-run")])
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    with pytest.raises(InvalidMessageState, match="reserved user message"):
        await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert assistant.status == "pending"
    assert model.calls == 0


@pytest.mark.asyncio
async def test_generate_persists_versioned_prompt_redacted_model_facts_and_null_reasoning(
    database, messages, saver
) -> None:
    """Catches audit persistence dropping approved facts or leaking adapter secrets."""
    context, state = await _reserved_context(database, messages, "audit")
    model = DeterministicModel(
        [
            ModelDelta(
                content="无 reasoning 回答",
                usage=TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12),
                finish_reason="length",
                response_metadata={
                    "provider_request": "ok",
                    "authorization": "Bearer provider-secret",
                },
            )
        ]
    )
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    prompt = json.loads(assistant.prompt_json or "null")
    parameters = json.loads(assistant.parameters_json or "null")
    trace = json.loads(assistant.trace_json)
    serialized = json.dumps(
        {"prompt": prompt, "parameters": parameters, "trace": trace},
        ensure_ascii=False,
    )
    assert prompt[0]["role"] == "system"
    assert "v1" in prompt[0]["content"]
    assert prompt[-1] == {"role": "user", "content": "来自 audit 的问题"}
    assert parameters == {
        "api_key": "[REDACTED]",
        "model": "test-model",
        "temperature": 0.2,
    }
    assert assistant.provider == "test-provider"
    assert assistant.model == "test-model"
    assert (assistant.input_tokens, assistant.output_tokens, assistant.total_tokens) == (
        8,
        4,
        12,
    )
    assert assistant.finish_reason == "length"
    assert assistant.reasoning_content is None
    assert trace["model_calls"][0]["response_metadata"] == {
        "authorization": "[REDACTED]",
        "provider_request": "ok",
    }
    assert "sk-test-secret" not in serialized
    assert "provider-secret" not in serialized


@pytest.mark.asyncio
async def test_generate_loads_only_recent_completed_context_for_its_conversation(
    database, messages, saver
) -> None:
    """Catches cross-conversation, failed-output, or unbounded context entering the prompt."""
    first_context, first_state = await _reserved_context(database, messages, "history-owner")
    first_model = DeterministicModel([ModelDelta(content="先前回答")])
    first_graph = compile_turn_graph(
        NodeDependencies(messages=messages, model=first_model), saver
    )
    await first_graph.ainvoke(first_state, _config(first_context), context=first_context)

    failed_turn = await messages.reserve_turn(
        first_context.conversation_id, str(uuid4()), "不应携带失败助手正文"
    )
    await messages.mark_streaming(failed_turn.assistant.id)
    await messages.flush_partial(
        failed_turn.assistant.id,
        content="失败秘密",
        reasoning_content=None,
        trace={"schema_version": 1, "nodes": []},
    )
    await messages.fail_assistant(
        failed_turn.assistant.id,
        error_code="model_error",
        error_message="safe",
    )
    await _reserved_context(database, messages, "other-owner")

    context, state = await _reserved_context(database, messages, "history-owner")
    model = DeterministicModel([ModelDelta(content="当前回答")])
    graph = compile_turn_graph(
        NodeDependencies(messages=messages, model=model, context_message_limit=2),
        saver,
    )

    await graph.ainvoke(state, _config(context), context=context)

    sent = [(message.type, message.content) for message in model.prompts[0]]
    assert sent[1:] == [
        ("human", "不应携带失败助手正文"),
        ("human", "来自 history-owner 的问题"),
    ]
    assert "失败秘密" not in repr(sent)
    assert "other-owner" not in repr(sent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "advances", "expected_flush"),
    [
        (["x" * 200, "y" * 56], [0.0, 0.0], "x" * 200 + "y" * 56),
        (["甲", "乙"], [0.0, 0.251], "甲乙"),
    ],
    ids=("256-new-characters", "250ms-elapsed"),
)
async def test_generate_flushes_partial_on_either_threshold(
    database, messages, saver, chunks, advances, expected_flush
) -> None:
    """Catches partial durability requiring both thresholds instead of either one."""
    context, state = await _reserved_context(database, messages, f"flush-{len(chunks[0])}")
    clock = ManualClock()
    recording = RecordingMessages(messages)
    model = DeterministicModel(
        [ModelDelta(content=chunk) for chunk in chunks],
        clock=clock,
        advances=advances,
    )
    graph = compile_turn_graph(
        NodeDependencies(messages=recording, model=model, clock=clock), saver
    )

    await graph.ainvoke(state, _config(context), context=context)

    assert recording.flushes == [(expected_flush, None)]


@pytest.mark.asyncio
async def test_model_failure_is_stable_retains_partials_and_is_idempotent(
    database, messages, saver
) -> None:
    """Catches raw provider failures leaking or checkpoint replay calling the model twice."""
    context, state = await _reserved_context(database, messages, "failure")
    model = DeterministicModel(
        [ModelDelta(content="部分", reasoning="真实部分依据")],
        error=RuntimeError("provider sk-live-very-secret"),
    )
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    first = await graph.ainvoke(state, _config(context), context=context)
    second = await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert first["phase"] == second["phase"] == "failed"
    assert first["error_code"] == second["error_code"] == "model_error"
    assert assistant.status == "failed"
    assert assistant.content == "部分"
    assert assistant.reasoning_content == "真实部分依据"
    assert assistant.error_message == "model generation failed"
    assert "sk-live-very-secret" not in assistant.trace_json
    assert model.calls == 1


@pytest.mark.asyncio
async def test_database_flush_failure_is_not_misreported_as_model_failure(
    database, messages, saver
) -> None:
    """Catches a business-store outage being swallowed and finalized as model_error."""
    context, state = await _reserved_context(database, messages, "database-failure")
    model = DeterministicModel([ModelDelta(content="x" * 256)])
    graph = compile_turn_graph(
        NodeDependencies(messages=FailingFlushMessages(messages), model=model), saver
    )

    with pytest.raises(RuntimeError, match="database write failed"):
        await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert assistant.status == "streaming"
    assert assistant.error_code is None


@pytest.mark.asyncio
async def test_completed_replay_does_not_call_model_or_overwrite_business_facts(
    database, messages, saver
) -> None:
    """Catches a completed checkpoint replay regenerating and replacing an answer."""
    context, state = await _reserved_context(database, messages, "replay")
    model = DeterministicModel([ModelDelta(content="原始回答")])
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)
    await graph.ainvoke(state, _config(context), context=context)
    original = await messages.find_assistant(context.conversation_id, context.request_id)

    model.deltas = [ModelDelta(content="覆盖回答")]
    replayed = await graph.ainvoke(state, _config(context), context=context)
    after = await messages.find_assistant(context.conversation_id, context.request_id)

    assert replayed["phase"] == "completed"
    assert after == original
    assert model.calls == 1


@pytest.mark.asyncio
async def test_orphaned_streaming_turn_fails_recovery_without_calling_model(
    database, messages, saver
) -> None:
    """Catches recovery of an active business fact restarting a non-idempotent model call."""
    context, state = await _reserved_context(database, messages, "orphaned")
    await messages.mark_streaming(context.assistant_message_id)
    await messages.flush_partial(
        context.assistant_message_id,
        content="已落库部分",
        reasoning_content="已落库依据",
        trace={"schema_version": 1, "nodes": []},
    )
    model = DeterministicModel([ModelDelta(content="重复生成")])
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    result = await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert result == {**state, "phase": "failed", "error_code": "process_interrupted"}
    assert assistant.status == "failed"
    assert assistant.content == "已落库部分"
    assert assistant.reasoning_content == "已落库依据"
    assert model.calls == 0


@pytest.mark.asyncio
async def test_publisher_failure_is_observable_but_does_not_break_business_completion(
    database, messages, saver
) -> None:
    """Catches a display-channel outage corrupting persisted message completion."""
    publisher = RecordingPublisher(fail=True)
    context, state = await _reserved_context(database, messages, "publisher", publisher)
    model = DeterministicModel([ModelDelta(content="仍然完成")])
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    result = await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    trace = json.loads(assistant.trace_json)
    assert result["phase"] == "completed"
    assert assistant.status == "completed"
    assert assistant.content == "仍然完成"
    assert trace["errors"] == [
        {"code": "event_publish_error", "message": "token event publication failed"}
    ]
    assert trace["stream"]["client_disconnected"] is True
    assert "publisher-secret" not in assistant.trace_json


@pytest.mark.asyncio
async def test_official_sqlite_saver_uses_same_file_and_isolates_thread_snapshots(
    database, messages, saver
) -> None:
    """Catches custom checkpoint storage, shared threads, or heavyweight state snapshots."""
    contexts: list[TurnContext] = []
    graph = compile_turn_graph(
        NodeDependencies(
            messages=messages,
            model=DeterministicModel([ModelDelta(content="隔离回答")]),
        ),
        saver,
    )
    for identifier in ("checkpoint-a", "checkpoint-b"):
        context, state = await _reserved_context(database, messages, identifier)
        contexts.append(context)
        await graph.ainvoke(state, _config(context), context=context)

    snapshots = [await graph.aget_state(_config(context)) for context in contexts]
    async with aiosqlite.connect(Path(database.path)) as connection:
        table_rows = await (
            await connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('messages', 'checkpoints', 'writes') ORDER BY name"
            )
        ).fetchall()
        checkpoint_rows = await (
            await connection.execute(
                "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
            )
        ).fetchall()
        writes_count = (
            await (await connection.execute("SELECT COUNT(*) FROM writes")).fetchone()
        )[0]

    assert [row[0] for row in table_rows] == ["checkpoints", "messages", "writes"]
    assert {row[0] for row in checkpoint_rows} == {
        contexts[0].thread_id,
        contexts[1].thread_id,
    }
    assert all(row[1] > 0 for row in checkpoint_rows)
    assert writes_count > 0
    assert all(set(snapshot.values) == STATE_FIELDS for snapshot in snapshots)
    assert snapshots[0].values["conversation_id"] != snapshots[1].values["conversation_id"]
    serialized = repr([snapshot.values for snapshot in snapshots])
    assert "sk-test-secret" not in serialized
    assert "BaseMessage" not in serialized
    assert "来自 checkpoint" not in serialized
