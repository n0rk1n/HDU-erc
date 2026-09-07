from __future__ import annotations

import asyncio
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
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from chatbot.core.errors import InvalidMessageState
from chatbot.db.messages import MessageRepository
from chatbot.graph import (
    EventPublisher,

    TurnContext,
    TurnState,
    build_turn_graph,
    compile_turn_graph,
)
from chatbot.graph.nodes import TurnNodes
from tests.emotion.helpers import dependencies as NodeDependencies
from chatbot.llm.types import ModelDelta, TokenUsage
from chatbot.services.events import TurnSubscription
from chatbot.services.identity import IdentityService


STATE_FIELDS = {
    "user_id",
    "conversation_id",
    "request_id",
    "user_message_id",
    "assistant_message_id",
    "emotion_analysis_id",
    "gate_decision_id",
    "gate_action",
    "emotion_status",
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

    async def publish(self, name: str, data: dict[str, object]) -> bool:
        if self.fail:
            raise RuntimeError("publisher-secret")
        self.events.append((name, data))
        return True


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


class FirstFlushFailsMessages(RecordingMessages):
    def __init__(self, wrapped: MessageRepository) -> None:
        super().__init__(wrapped)
        self.attempts = 0

    async def flush_partial(self, assistant_message_id: str, **facts: object):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("database-secret first flush failed")
        return await self.wrapped.flush_partial(assistant_message_id, **facts)


class FailureWriteFailsMessages(RecordingMessages):
    async def fail_assistant(self, assistant_message_id: str, **facts: object):
        raise RuntimeError("database-secret failure write failed")


class CompletionFailsMessages(RecordingMessages):
    async def complete_assistant(self, assistant_message_id: str, **facts: object):
        raise RuntimeError("database-secret completion failed")


class CancelledModel(DeterministicModel):
    async def stream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[ModelDelta]:
        self.calls += 1
        self.prompts.append(prompt)
        yield ModelDelta(content="取消前正文", reasoning="取消前依据")
        raise asyncio.CancelledError


@pytest_asyncio.fixture
async def messages(database) -> MessageRepository:
    return MessageRepository(database)


@pytest_asyncio.fixture
async def saver(database):
    async with AsyncSqliteSaver.from_conn_string(str(database.path)) as checkpointer:
        await checkpointer.conn.execute("PRAGMA journal_mode = WAL")
        await checkpointer.conn.execute("PRAGMA foreign_keys = ON")
        await checkpointer.conn.execute("PRAGMA busy_timeout = 5000")
        await checkpointer.conn.execute("PRAGMA synchronous = NORMAL")
        await checkpointer.conn.commit()
        await checkpointer.setup()
        yield checkpointer


async def _reserved_context(
    database,
    messages: MessageRepository,
    identifier: str,
    publisher: EventPublisher | None = None,
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
        "decide_emotion",
        "analyze_emotion",
        "generate_response",
        "finalize_turn",
    ]
    assert [node["status"] for node in trace["nodes"]] == [
        "completed",
        "completed",
        "failed",
        "completed",
        "completed",
    ]
    assert assistant.content == "答案"
    assert assistant.reasoning_content == "真实依据"
    assert [event for event in context.publisher.events if event[0] == "token"] == [
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
async def test_thread_config_mismatch_is_rejected_before_checkpoint_or_business_write(
    database, messages, saver
) -> None:
    """Catches context A mutating its message while recording progress under thread B."""
    context_a, state_a = await _reserved_context(database, messages, "thread-context-a")
    context_b, _ = await _reserved_context(database, messages, "thread-config-b")
    model = DeterministicModel([ModelDelta(content="must-not-run")])
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    assert isinstance(graph, CompiledStateGraph)
    with pytest.raises(InvalidMessageState, match="thread_id"):
        await graph.ainvoke(state_a, _config(context_b), context=context_a)
    with pytest.raises(InvalidMessageState, match="thread_id"):
        graph.astream(state_a, _config(context_b), context=context_a)

    assistant_a = await messages.find_assistant(
        context_a.conversation_id, context_a.request_id
    )
    async with aiosqlite.connect(Path(database.path)) as connection:
        b_checkpoints = (
            await (
                await connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    (context_b.thread_id,),
                )
            ).fetchone()
        )[0]

    assert assistant_a is not None
    assert assistant_a.status == "pending"
    assert model.calls == 0
    assert b_checkpoints == 0


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
    from chatbot.llm.prompt import get_system_prompt
    assert prompt[0]["content"] == get_system_prompt()
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
async def test_graph_never_persists_extended_credentials_or_authenticated_urls(
    database, messages, saver
) -> None:
    """Catches secrets reaching Message business columns or trace through model facts."""
    context, state = await _reserved_context(database, messages, "credential-audit")
    model = DeterministicModel(
        [
            ModelDelta(
                content="安全回答",
                usage=TokenUsage(input_tokens=9, output_tokens=4, total_tokens=13),
                response_metadata={
                    "PASSWORD": "metadata-password-secret",
                    "nested": {"client-secret": "metadata-client-secret"},
                    "token_usage": {"total_tokens": 13},
                },
            )
        ]
    )
    model.parameters = {
        "model": "test-model",
        "passwd": "parameter-password-secret",
        "X_API_KEY": "parameter-api-secret",
        "refresh_token": "parameter-refresh-secret",
        "private-key": "parameter-private-secret",
        "base_url": (
            "https://user:parameter-url-secret@example.invalid/v1"
            "?client_secret=query-secret&region=cn"
        ),
        "token_usage": 13,
    }
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    parameters = json.loads(assistant.parameters_json or "null")
    trace = json.loads(assistant.trace_json)
    assert parameters == {
        "model": "test-model",
        "passwd": "[REDACTED]",
        "X_API_KEY": "[REDACTED]",
        "refresh_token": "[REDACTED]",
        "private-key": "[REDACTED]",
        "base_url": (
            "https://example.invalid/v1?client_secret=[REDACTED]&region=cn"
        ),
        "token_usage": 13,
    }
    assert trace["model_calls"][0]["response_metadata"] == {
        "PASSWORD": "[REDACTED]",
        "nested": {"client-secret": "[REDACTED]"},
        "token_usage": {"total_tokens": 13},
    }
    serialized_business_facts = "|".join(
        value
        for value in (
            assistant.trace_json,
            assistant.prompt_json,
            assistant.parameters_json,
            assistant.error_message,
        )
        if value is not None
    )
    for secret in (
        "metadata-password-secret",
        "metadata-client-secret",
        "parameter-password-secret",
        "parameter-api-secret",
        "parameter-refresh-secret",
        "parameter-private-secret",
        "parameter-url-secret",
        "query-secret",
    ):
        assert secret not in serialized_business_facts


@pytest.mark.asyncio
async def test_usage_snapshots_keep_each_fields_last_known_value(
    database, messages, saver
) -> None:
    """Catches a later partial cumulative usage snapshot clearing known token counts."""
    context, state = await _reserved_context(database, messages, "usage-snapshots")
    model = DeterministicModel(
        [
            ModelDelta(content="答", usage=TokenUsage(input_tokens=10)),
            ModelDelta(
                content="案",
                usage=TokenUsage(output_tokens=4, total_tokens=14),
            ),
            ModelDelta(usage=TokenUsage(), finish_reason="stop"),
        ]
    )
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=model), saver)

    await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert (assistant.input_tokens, assistant.output_tokens, assistant.total_tokens) == (
        10,
        4,
        14,
    )
    assert json.loads(assistant.trace_json)["model_calls"][0]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }


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

    assert recording.flushes == [
        (expected_flush, None),
        (expected_flush, None),
    ]


@pytest.mark.asyncio
async def test_short_generation_survives_rebuild_before_finalize(
    database, messages, saver
) -> None:
    """Catches sub-threshold final chunks existing only in process-local run facts."""
    context, state = await _reserved_context(database, messages, "short-recovery")
    first_model = DeterministicModel(
        [ModelDelta(content="短答", reasoning="短依据")]
    )
    interrupted = build_turn_graph(
        NodeDependencies(messages=messages, model=first_model)
    ).compile(checkpointer=saver, interrupt_after=["generate_response"])

    generated = await interrupted.ainvoke(state, _config(context), context=context)
    before_rebuild = await messages.find_assistant(
        context.conversation_id, context.request_id
    )

    replacement_model = DeterministicModel([ModelDelta(content="不得重试")])
    rebuilt = compile_turn_graph(
        NodeDependencies(messages=messages, model=replacement_model), saver
    )
    recovered = await rebuilt.ainvoke(None, _config(context), context=context)
    after_rebuild = await messages.find_assistant(
        context.conversation_id, context.request_id
    )

    assert generated["phase"] == "generated"
    assert before_rebuild is not None
    assert before_rebuild.status == "streaming"
    assert before_rebuild.content == "短答"
    assert before_rebuild.reasoning_content == "短依据"
    assert after_rebuild is not None
    assert recovered["phase"] == "failed"
    assert recovered["error_code"] == "process_interrupted"
    assert after_rebuild.status == "failed"
    assert after_rebuild.content == "短答"
    assert after_rebuild.reasoning_content == "短依据"
    assert first_model.calls == 1
    assert replacement_model.calls == 0


@pytest.mark.asyncio
async def test_model_failure_is_stable_retains_partials_and_is_idempotent(
    database, messages, saver
) -> None:
    """Catches raw provider failures leaking or checkpoint replay calling the model twice."""
    context, state = await _reserved_context(database, messages, "failure")
    clock = ManualClock()
    model = DeterministicModel(
        [
            ModelDelta(
                content="部分",
                reasoning="真实部分依据",
                usage=TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                response_metadata={
                    "provider_request_id": "known-id",
                    "authorization": "Bearer provider-secret",
                },
            )
        ],
        error=RuntimeError("provider sk-live-very-secret"),
        clock=clock,
        advances=[0.125],
    )
    graph = compile_turn_graph(
        NodeDependencies(messages=messages, model=model, clock=clock), saver
    )

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
    assert assistant.provider == "test-provider"
    assert assistant.model == "test-model"
    assert json.loads(assistant.parameters_json or "null") == {
        "api_key": "[REDACTED]",
        "model": "test-model",
        "temperature": 0.2,
    }
    prompt = json.loads(assistant.prompt_json or "null")
    assert prompt[0]["role"] == "system"
    assert prompt[-1] == {"role": "user", "content": "来自 failure 的问题"}
    assert (assistant.input_tokens, assistant.output_tokens, assistant.total_tokens) == (
        5,
        2,
        7,
    )
    assert assistant.latency_ms == 125
    assert assistant.finish_reason == "error"
    assert json.loads(assistant.trace_json)["model_calls"][0][
        "response_metadata"
    ] == {
        "authorization": "[REDACTED]",
        "provider_request_id": "known-id",
    }
    serialized = "|".join(
        filter(
            None,
            (
                assistant.trace_json,
                assistant.prompt_json,
                assistant.parameters_json,
                assistant.error_message,
            ),
        )
    )
    assert "sk-live-very-secret" not in serialized
    assert "provider-secret" not in serialized
    assert model.calls == 1


@pytest.mark.asyncio
async def test_database_flush_failure_reaches_stable_failed_terminal_state(
    database, messages, saver
) -> None:
    """Catches a business-store outage being swallowed and finalized as model_error."""
    context, state = await _reserved_context(database, messages, "database-failure")
    model = DeterministicModel([ModelDelta(content="x" * 256)])
    graph = compile_turn_graph(
        NodeDependencies(messages=FailingFlushMessages(messages), model=model), saver
    )

    result = await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert result["phase"] == "failed"
    assert result["error_code"] == "database_error"
    assert assistant.status == "failed"
    assert assistant.error_code == "database_error"


@pytest.mark.asyncio
async def test_flush_failure_best_effort_persists_all_run_facts_and_clears_memory(
    database, messages
) -> None:
    """Catches infrastructure failure losing received facts or retaining process secrets."""
    context, state = await _reserved_context(database, messages, "flush-facts")
    wrapped = FirstFlushFailsMessages(messages)
    model = DeterministicModel(
        [
            ModelDelta(
                content="已接收正文",
                reasoning="已接收依据",
                usage=TokenUsage(input_tokens=11, output_tokens=5, total_tokens=16),
                finish_reason="stop",
                response_metadata={"client_secret": "metadata-secret", "request_id": "safe"},
            )
        ]
    )
    model.parameters = {
        "model": "test-model",
        "password": "parameter-secret",
        "temperature": 0.2,
    }
    nodes = TurnNodes(
        NodeDependencies(
            messages=wrapped,
            model=model,
            flush_character_threshold=1,
        )
    )
    runtime = Runtime(context=context)

    prepared = {**state, **(await nodes.prepare_turn(state, runtime))}
    result = await nodes.generate_response(prepared, runtime)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert result == {"phase": "failed", "error_code": "database_error"}
    assert assistant.status == "failed"
    assert assistant.error_code == "database_error"
    assert assistant.error_message == "database operation failed"
    assert assistant.content == "已接收正文"
    assert assistant.reasoning_content == "已接收依据"
    assert json.loads(assistant.prompt_json or "null")[-1] == {
        "role": "user",
        "content": "来自 flush-facts 的问题",
    }
    assert json.loads(assistant.parameters_json or "null") == {
        "model": "test-model",
        "password": "[REDACTED]",
        "temperature": 0.2,
    }
    assert (assistant.input_tokens, assistant.output_tokens, assistant.total_tokens) == (
        11,
        5,
        16,
    )
    trace = json.loads(assistant.trace_json)
    assert trace["model_calls"][0]["response_metadata"] == {
        "client_secret": "[REDACTED]",
        "request_id": "safe",
    }
    assert trace["errors"][-1] == {
        "code": "database_error",
        "message": "database operation failed",
    }
    assert context.assistant_message_id not in nodes._runs
    assert "database-secret" not in "|".join(
        filter(None, (assistant.trace_json, assistant.prompt_json, assistant.parameters_json, assistant.error_message))
    )


@pytest.mark.asyncio
async def test_cancellation_propagates_and_clears_run_facts_even_if_failure_write_fails(
    database, messages
) -> None:
    """Catches cancellation being swallowed or _RunFacts surviving failed cleanup."""
    context, state = await _reserved_context(database, messages, "cancel-cleanup")
    nodes = TurnNodes(
        NodeDependencies(
            messages=FailureWriteFailsMessages(messages),
            model=CancelledModel([]),
        )
    )
    runtime = Runtime(context=context)
    prepared = {**state, **(await nodes.prepare_turn(state, runtime))}

    with pytest.raises(asyncio.CancelledError):
        await nodes.generate_response(prepared, runtime)

    assert context.assistant_message_id not in nodes._runs


@pytest.mark.asyncio
async def test_completion_write_failure_uses_graph_facts_for_failed_terminal_state(
    database, messages
) -> None:
    """Catches final business transaction failure falling back to partial-only facts."""
    context, state = await _reserved_context(database, messages, "complete-facts")
    model = DeterministicModel(
        [
            ModelDelta(
                content="完成前正文",
                reasoning="完成前依据",
                usage=TokenUsage(input_tokens=6, output_tokens=3, total_tokens=9),
                response_metadata={"private_key": "metadata-secret"},
            )
        ]
    )
    nodes = TurnNodes(
        NodeDependencies(messages=CompletionFailsMessages(messages), model=model)
    )
    runtime = Runtime(context=context)
    prepared = {**state, **(await nodes.prepare_turn(state, runtime))}
    generated = {**prepared, **(await nodes.generate_response(prepared, runtime))}

    result = await nodes.finalize_turn(generated, runtime)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert result == {"phase": "failed", "error_code": "database_error"}
    assert assistant.status == "failed"
    assert assistant.content == "完成前正文"
    assert assistant.reasoning_content == "完成前依据"
    assert json.loads(assistant.prompt_json or "null")[-1]["content"] == "来自 complete-facts 的问题"
    assert (assistant.input_tokens, assistant.output_tokens, assistant.total_tokens) == (6, 3, 9)
    assert "metadata-secret" not in assistant.trace_json
    assert context.assistant_message_id not in nodes._runs


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
async def test_real_subscription_overflow_sets_client_disconnected_without_breaking_completion(
    database, messages, saver
) -> None:
    """Catches real queue detachment being absent from the persisted stream trace."""
    subscription = TurnSubscription(queue_capacity=1)
    assert await subscription.publish("run_started", {"request_id": "occupied"}) is True
    context, state = await _reserved_context(
        database, messages, "real-overflow", subscription
    )
    graph = compile_turn_graph(
        NodeDependencies(
            messages=messages,
            model=DeterministicModel([ModelDelta(content="后台仍完成")]),
        ),
        saver,
    )

    result = await graph.ainvoke(state, _config(context), context=context)

    assistant = await messages.find_assistant(context.conversation_id, context.request_id)
    assert assistant is not None
    assert result["phase"] == "completed"
    assert assistant.status == "completed"
    assert assistant.content == "后台仍完成"
    assert json.loads(assistant.trace_json)["stream"]["client_disconnected"] is True


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
