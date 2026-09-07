from __future__ import annotations

import asyncio
from collections.abc import Mapping
from uuid import uuid4

import pytest

from chatbot.core.errors import (
    DatabaseError,
    InvalidMessageState,
    ProcessInterrupted,
    TurnInProgress,
    UserNotFound,
)
from chatbot.db.conversations import ConversationRepository
from chatbot.db.messages import MessageRepository
from chatbot.graph.state import TurnContext
from chatbot.services.events import SseEvent, TurnSubscription
from chatbot.services.identity import IdentityService
from chatbot.services.turns import TurnCoordinator


class ScriptedGraph:
    """Exercise the coordinator boundary while keeping repository effects real."""

    def __init__(
        self,
        messages: MessageRepository,
        *,
        tokens: tuple[str, ...] = ("答",),
        block: bool = False,
        fail: bool = False,
    ) -> None:
        self.messages = messages
        self.tokens = tokens
        self.block = block
        self.fail = fail
        self.calls: list[tuple[object, object, TurnContext]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(
        self,
        state: object,
        config: object = None,
        *,
        context: TurnContext,
        **_: object,
    ) -> dict[str, object]:
        self.calls.append((state, config, context))
        assert isinstance(state, Mapping)
        assert config == {"configurable": {"thread_id": context.thread_id}}
        assert state["request_id"] == context.request_id
        await self.messages.mark_streaming(context.assistant_message_id)
        self.entered.set()
        if self.block:
            await self.release.wait()

        content = ""
        for token in self.tokens:
            content += token
            await context.publisher.publish("token", {"content": token})

        if self.fail:
            await self.messages.fail_assistant(
                context.assistant_message_id,
                error_code="model_error",
                error_message="model generation failed",
                content=content,
                trace={"private": "trace"},
                prompt={"private": "prompt"},
                parameters={"private": "parameters"},
            )
            return {"phase": "failed", "error_code": "model_error"}

        await self.messages.complete_assistant(
            context.assistant_message_id,
            content=content,
            reasoning_content="private reasoning",
            trace={"private": "trace"},
            prompt={"private": "prompt"},
            provider="test",
            model="scripted",
            parameters={"private": "parameters"},
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=1,
            finish_reason="stop",
        )
        return {"phase": "completed", "error_code": None}


class ConcurrentGraph(ScriptedGraph):
    def __init__(self, messages: MessageRepository) -> None:
        super().__init__(messages)
        self.entered_count = 0
        self.all_entered = asyncio.Event()

    async def ainvoke(
        self,
        state: object,
        config: object = None,
        *,
        context: TurnContext,
        **kwargs: object,
    ) -> dict[str, object]:
        self.entered_count += 1
        if self.entered_count == 2:
            self.all_entered.set()
        await asyncio.wait_for(self.all_entered.wait(), timeout=1)
        return await super().ainvoke(
            state, config, context=context, **kwargs
        )


async def _resolved(database, identifier: str):
    return await IdentityService(database).resolve(identifier)


def _coordinator(database, messages, graph, *, queue_capacity: int = 64):
    return TurnCoordinator(
        conversations=ConversationRepository(database),
        messages=messages,
        graph=graph,
        queue_capacity=queue_capacity,
    )


async def _collect(subscription: TurnSubscription) -> list[SseEvent]:
    return [event async for event in subscription.events()]


@pytest.mark.asyncio
async def test_done_is_emitted_after_database_completion(database) -> None:
    """Catches done being published from graph output before the committed fact is read."""
    identity = await _resolved(database, "committed")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, tokens=("答", "案"))
    coordinator = _coordinator(database, messages, graph)
    request_id = str(uuid4())

    events = await _collect(
        await coordinator.open_stream(identity.user.id, request_id, "你好")
    )
    assistant = await messages.find_assistant(identity.conversation.id, request_id)

    assert assistant is not None
    assert assistant.status == "completed"
    assert [event.name for event in events] == [
        "run_started",
        "user_message",
        "token",
        "token",
        "done",
    ]
    assert events[-1].data["message"] == {
        "id": assistant.id,
        "request_id": request_id,
        "sequence_no": assistant.sequence_no,
        "role": "assistant",
        "status": "completed",
        "content": "答案",
        "error_code": None,
        "error_message": None,
        "created_at": assistant.created_at,
        "updated_at": assistant.updated_at,
        "completed_at": assistant.completed_at,
        "processing": None,
    }
    assert events[-1].data["replayed"] is False
    assert not ({"reasoning_content", "trace_json", "prompt_json", "parameters_json", "thread_id"} & events[-1].data["message"].keys())


@pytest.mark.asyncio
async def test_same_user_different_request_is_rejected_immediately(database) -> None:
    """Catches a second request waiting on the user lock or entering the graph."""
    identity = await _resolved(database, "locked")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, block=True)
    coordinator = _coordinator(database, messages, graph)
    first = await coordinator.open_stream(identity.user.id, str(uuid4()), "first")
    await asyncio.wait_for(graph.entered.wait(), timeout=1)

    with pytest.raises(TurnInProgress) as error:
        await asyncio.wait_for(
            coordinator.open_stream(identity.user.id, str(uuid4()), "second"),
            timeout=0.1,
        )

    assert error.value.code == "turn_in_progress"
    assert len(graph.calls) == 1
    graph.release.set()
    await _collect(first)


@pytest.mark.asyncio
async def test_different_users_reach_graph_concurrently(database) -> None:
    """Catches a global generation lock serializing independent users."""
    alice = await _resolved(database, "alice")
    bob = await _resolved(database, "bob")
    messages = MessageRepository(database)
    graph = ConcurrentGraph(messages)
    coordinator = _coordinator(database, messages, graph)

    first = await coordinator.open_stream(alice.user.id, str(uuid4()), "one")
    second = await coordinator.open_stream(bob.user.id, str(uuid4()), "two")

    await asyncio.wait_for(graph.all_entered.wait(), timeout=1)
    first_events, second_events = await asyncio.gather(_collect(first), _collect(second))
    assert [events[-1].name for events in (first_events, second_events)] == [
        "done",
        "done",
    ]
    assert {call[2].user_id for call in graph.calls} == {alice.user.id, bob.user.id}


@pytest.mark.asyncio
async def test_completed_request_replays_during_another_active_turn(database) -> None:
    """Catches completed idempotent replay being rejected by an unrelated active lock."""
    identity = await _resolved(database, "replay")
    messages = MessageRepository(database)
    first_graph = ScriptedGraph(messages, tokens=("旧",))
    first_coordinator = _coordinator(database, messages, first_graph)
    completed_request = str(uuid4())
    await _collect(
        await first_coordinator.open_stream(
            identity.user.id, completed_request, "saved"
        )
    )

    active_graph = ScriptedGraph(messages, tokens=("新",), block=True)
    coordinator = _coordinator(database, messages, active_graph)
    active = await coordinator.open_stream(identity.user.id, str(uuid4()), "active")
    await asyncio.wait_for(active_graph.entered.wait(), timeout=1)

    replay = await _collect(
        await coordinator.open_stream(identity.user.id, completed_request, "saved")
    )

    assert [event.name for event in replay] == ["run_started", "done"]
    assert replay[-1].data["replayed"] is True
    assert len(active_graph.calls) == 1
    active_graph.release.set()
    await _collect(active)


@pytest.mark.asyncio
async def test_changed_content_cannot_reuse_a_completed_request(database) -> None:
    """Catches request IDs replaying a different user input."""
    identity = await _resolved(database, "replay-content")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages)
    coordinator = _coordinator(database, messages, graph)
    request_id = str(uuid4())
    await _collect(await coordinator.open_stream(identity.user.id, request_id, "one"))

    with pytest.raises(InvalidMessageState) as error:
        await coordinator.open_stream(identity.user.id, request_id, "two")

    assert error.value.code == "invalid_message"
    assert len(graph.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "streaming"])
async def test_nonterminal_existing_request_is_never_started_again(database, status) -> None:
    """Catches stale pending or streaming facts being mistaken for a new turn."""
    identity = await _resolved(database, f"stale-{status}")
    messages = MessageRepository(database)
    request_id = str(uuid4())
    turn = await messages.reserve_turn(identity.conversation.id, request_id, "saved")
    if status == "streaming":
        await messages.mark_streaming(turn.assistant.id)
    graph = ScriptedGraph(messages)
    coordinator = _coordinator(database, messages, graph)

    with pytest.raises(TurnInProgress) as error:
        await coordinator.open_stream(identity.user.id, request_id, "saved")

    assert error.value.code == "turn_in_progress"
    assert graph.calls == []


@pytest.mark.asyncio
async def test_failed_request_replays_only_a_stable_public_error(database) -> None:
    """Catches failed replay leaking stored private fields or raw error details."""
    identity = await _resolved(database, "failed-replay")
    messages = MessageRepository(database)
    request_id = str(uuid4())
    turn = await messages.reserve_turn(identity.conversation.id, request_id, "saved")
    await messages.mark_streaming(turn.assistant.id)
    await messages.fail_assistant(
        turn.assistant.id,
        error_code="model_error",
        error_message="secret upstream exception: sk-private",
        content="partial",
        reasoning_content="private reasoning",
        trace={"authorization": "private"},
        prompt={"secret": "private"},
        parameters={"api_key": "private"},
    )
    graph = ScriptedGraph(messages)
    coordinator = _coordinator(database, messages, graph)

    events = await _collect(
        await coordinator.open_stream(identity.user.id, request_id, "saved")
    )

    assert events == [
        SseEvent(
            name="error",
            data={"code": "model_error", "message": "model generation failed"},
        )
    ]
    assert graph.calls == []


@pytest.mark.asyncio
async def test_failed_task_publishes_public_error_and_releases_user_activity(database) -> None:
    """Catches failed tasks leaking private facts or keeping their user lock forever."""
    identity = await _resolved(database, "failed-active")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, tokens=("partial",), fail=True)
    coordinator = _coordinator(database, messages, graph)

    first = await _collect(
        await coordinator.open_stream(identity.user.id, str(uuid4()), "first")
    )
    second = await _collect(
        await coordinator.open_stream(identity.user.id, str(uuid4()), "second")
    )

    assert [event.name for event in first] == [
        "run_started",
        "user_message",
        "token",
        "error",
    ]
    assert first[-1].data == {
        "code": "model_error",
        "message": "model generation failed",
    }
    assert "private" not in repr(first[-1].data)
    assert second[-1].name == "error"
    assert len(graph.calls) == 2


@pytest.mark.asyncio
async def test_detach_discards_display_events_without_cancelling_generation(database) -> None:
    """Catches browser disconnect cancelling or accumulating the background producer."""
    identity = await _resolved(database, "detached")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, tokens=("a", "b", "c"), block=True)
    coordinator = _coordinator(database, messages, graph, queue_capacity=3)
    request_id = str(uuid4())
    subscription = await coordinator.open_stream(identity.user.id, request_id, "hello")
    await asyncio.wait_for(graph.entered.wait(), timeout=1)

    subscription.detach()
    graph.release.set()
    await coordinator.shutdown(timeout_seconds=1)

    assistant = await messages.find_assistant(identity.conversation.id, request_id)
    assert assistant is not None
    assert assistant.status == "completed"
    assert assistant.content == "abc"
    assert await _collect(subscription) == []


@pytest.mark.asyncio
async def test_queue_overflow_detaches_slow_consumer_and_never_blocks_completion(database) -> None:
    """Catches an unbounded or blocking browser queue stalling persisted completion."""
    identity = await _resolved(database, "slow")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, tokens=tuple(str(i) for i in range(100)))
    coordinator = _coordinator(database, messages, graph, queue_capacity=2)
    request_id = str(uuid4())

    subscription = await coordinator.open_stream(identity.user.id, request_id, "hello")
    await coordinator.shutdown(timeout_seconds=1)

    assistant = await messages.find_assistant(identity.conversation.id, request_id)
    assert assistant is not None
    assert assistant.status == "completed"
    assert assistant.content == "".join(str(i) for i in range(100))
    assert await _collect(subscription) == []


@pytest.mark.asyncio
async def test_invalid_request_is_rejected_before_repository_access() -> None:
    """Catches malformed request facts reaching SQLite or surfacing raw DB failures."""
    class BombRepository:
        async def get_default_by_user(self, user_id: int):
            raise AssertionError(f"repository called for {user_id}")

    coordinator = TurnCoordinator(
        conversations=BombRepository(),
        messages=BombRepository(),
        graph=object(),
    )

    for request_id, content in (("not-a-uuid", "hello"), (str(uuid4()), " \n ")):
        with pytest.raises(InvalidMessageState) as error:
            await coordinator.open_stream(1, request_id, content)
        assert error.value.code == "invalid_message"


@pytest.mark.asyncio
async def test_missing_default_conversation_is_a_stable_domain_error(database) -> None:
    """Catches absent internal users/default conversations becoming attribute errors."""
    messages = MessageRepository(database)
    coordinator = _coordinator(database, messages, ScriptedGraph(messages))

    with pytest.raises(UserNotFound) as error:
        await coordinator.open_stream(999_999, str(uuid4()), "hello")

    assert error.value.code == "user_not_found"


@pytest.mark.asyncio
async def test_repository_failure_is_normalized_without_raw_details() -> None:
    """Catches database implementation errors escaping the service boundary."""
    class BrokenConversations:
        async def get_default_by_user(self, user_id: int):
            raise RuntimeError(f"private database details for {user_id}")

    coordinator = TurnCoordinator(
        conversations=BrokenConversations(),
        messages=object(),
        graph=object(),
    )

    with pytest.raises(DatabaseError) as error:
        await coordinator.open_stream(1, str(uuid4()), "hello")

    assert error.value.code == "database_error"
    assert "private" not in error.value.message


@pytest.mark.asyncio
async def test_task_creation_failure_finalizes_reserved_turn_for_failed_replay(
    database, monkeypatch
) -> None:
    """Catches startup failure leaving a committed assistant pending forever."""
    identity = await _resolved(database, "task-create-failure")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages)
    coordinator = _coordinator(database, messages, graph)
    request_id = str(uuid4())

    def fail_create_task(coroutine, **_: object):
        coroutine.close()
        raise RuntimeError("private task factory failure")

    monkeypatch.setattr(asyncio, "create_task", fail_create_task)
    failure: BaseException | None = None
    try:
        await coordinator.open_stream(identity.user.id, request_id, "hello")
    except BaseException as error:
        failure = error

    assistant = await messages.find_assistant(identity.conversation.id, request_id)
    assert assistant is not None
    assert assistant.status == "failed"
    assert assistant.error_code == "process_interrupted"
    assert isinstance(failure, DatabaseError)
    assert "private" not in failure.message

    replay = await _collect(
        await coordinator.open_stream(identity.user.id, request_id, "hello")
    )
    assert replay == [
        SseEvent(
            name="error",
            data={
                "code": "process_interrupted",
                "message": "generation interrupted",
            },
        )
    ]
    assert graph.calls == []


@pytest.mark.asyncio
async def test_second_cancellation_during_initialization_cleanup_releases_user_lock(
    database, monkeypatch
) -> None:
    """Catches cleanup cancellation skipping detach and per-user lock release."""
    identity = await _resolved(database, "cleanup-cancel")
    messages = MessageRepository(database)

    class CleanupBlockingMessages:
        def __init__(self) -> None:
            self.cleanup_started = asyncio.Event()
            self.block_cleanup = True

        def __getattr__(self, name: str):
            return getattr(messages, name)

        async def find_assistant(self, conversation_id: str, request_id: str):
            if self.block_cleanup:
                self.cleanup_started.set()
                await asyncio.Event().wait()
            return await messages.find_assistant(conversation_id, request_id)

    blocking_messages = CleanupBlockingMessages()
    graph = ScriptedGraph(messages)
    coordinator = TurnCoordinator(
        conversations=ConversationRepository(database),
        messages=blocking_messages,
        graph=graph,
    )
    original_create_task = asyncio.create_task
    failed_once = False

    def fail_first_producer(coroutine, **kwargs: object):
        nonlocal failed_once
        if not failed_once and str(kwargs.get("name", "")).startswith("chat-turn-"):
            failed_once = True
            coroutine.close()
            raise RuntimeError("private task factory failure")
        return original_create_task(coroutine, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_first_producer)
    request_id = str(uuid4())
    opening = original_create_task(
        coordinator.open_stream(identity.user.id, request_id, "first")
    )
    await asyncio.wait_for(blocking_messages.cleanup_started.wait(), timeout=1)

    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening

    interrupted = await messages.find_assistant(identity.conversation.id, request_id)
    assert interrupted is not None
    assert interrupted.status == "pending"

    blocking_messages.block_cleanup = False
    next_request = str(uuid4())
    events = await _collect(
        await coordinator.open_stream(identity.user.id, next_request, "second")
    )
    assert events[-1].name == "done"
    assert len(graph.calls) == 1


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_turn(database) -> None:
    """Catches graceful shutdown cancelling work that completes within its deadline."""
    identity = await _resolved(database, "shutdown-wait")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, block=True)
    coordinator = _coordinator(database, messages, graph)
    request_id = str(uuid4())
    subscription = await coordinator.open_stream(identity.user.id, request_id, "hello")
    await asyncio.wait_for(graph.entered.wait(), timeout=1)

    shutdown = asyncio.create_task(coordinator.shutdown(timeout_seconds=1))
    await asyncio.sleep(0)
    assert not shutdown.done()
    graph.release.set()
    await shutdown

    events = await _collect(subscription)
    assistant = await messages.find_assistant(identity.conversation.id, request_id)
    assert assistant is not None and assistant.status == "completed"
    assert events[-1].name == "done"


@pytest.mark.asyncio
async def test_shutdown_timeout_marks_cancelled_turn_failed(database) -> None:
    """Catches timeout cancellation leaving a pending/streaming assistant or lock."""
    identity = await _resolved(database, "shutdown-timeout")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, block=True)
    coordinator = _coordinator(database, messages, graph)
    request_id = str(uuid4())
    subscription = await coordinator.open_stream(identity.user.id, request_id, "hello")
    await asyncio.wait_for(graph.entered.wait(), timeout=1)

    await coordinator.shutdown(timeout_seconds=0.01)

    events = await _collect(subscription)
    assistant = await messages.find_assistant(identity.conversation.id, request_id)
    assert assistant is not None
    assert assistant.status == "failed"
    assert assistant.error_code == "process_interrupted"
    assert events[-1] == SseEvent(
        name="error",
        data={"code": "process_interrupted", "message": "generation interrupted"},
    )


@pytest.mark.asyncio
async def test_immediate_shutdown_fails_a_task_cancelled_before_it_starts(database) -> None:
    """Catches pre-start task cancellation bypassing the producer's cleanup block."""
    identity = await _resolved(database, "shutdown-before-start")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, block=True)
    coordinator = _coordinator(database, messages, graph)
    request_id = str(uuid4())
    subscription = await coordinator.open_stream(identity.user.id, request_id, "hello")

    await coordinator.shutdown(timeout_seconds=0)

    assistant = await messages.find_assistant(identity.conversation.id, request_id)
    assert assistant is not None
    assert assistant.status == "failed"
    assert assistant.error_code == "process_interrupted"
    events = await asyncio.wait_for(_collect(subscription), timeout=1)
    assert events[-1] == SseEvent(
        name="error",
        data={"code": "process_interrupted", "message": "generation interrupted"},
    )


@pytest.mark.asyncio
async def test_cancelling_shutdown_does_not_cancel_inflight_turn(database) -> None:
    """Catches caller cancellation being swallowed or forwarded into producers."""
    identity = await _resolved(database, "shutdown-caller-cancel")
    messages = MessageRepository(database)
    graph = ScriptedGraph(messages, block=True)
    coordinator = _coordinator(database, messages, graph)
    request_id = str(uuid4())
    subscription = await coordinator.open_stream(identity.user.id, request_id, "hello")
    await asyncio.wait_for(graph.entered.wait(), timeout=1)
    shutdown = asyncio.create_task(coordinator.shutdown(timeout_seconds=10))
    await asyncio.sleep(0)

    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown

    assistant = await messages.find_assistant(identity.conversation.id, request_id)
    assert assistant is not None and assistant.status == "streaming"
    graph.release.set()
    await coordinator.shutdown(timeout_seconds=1)
    events = await _collect(subscription)
    assert events[-1].name == "done"


@pytest.mark.asyncio
async def test_new_streams_are_rejected_after_shutdown_begins(database) -> None:
    """Catches shutdown racing with newly accepted background work."""
    identity = await _resolved(database, "shutdown-closed")
    messages = MessageRepository(database)
    coordinator = _coordinator(database, messages, ScriptedGraph(messages))
    await coordinator.shutdown(timeout_seconds=0)

    with pytest.raises(ProcessInterrupted) as error:
        await coordinator.open_stream(identity.user.id, str(uuid4()), "hello")

    assert error.value.code == "process_interrupted"


def test_event_subscription_rejects_nonpositive_capacity() -> None:
    """Catches invalid queue bounds silently creating blocking/unbounded queues."""
    with pytest.raises(ValueError, match="queue_capacity"):
        TurnSubscription(queue_capacity=0)


@pytest.mark.asyncio
async def test_event_subscription_reports_delivery_and_real_overflow_detachment() -> None:
    """Catches overflow silently looking like a delivered display event."""
    subscription = TurnSubscription(queue_capacity=1)

    first = await subscription.publish("run_started", {"request_id": "request-1"})
    overflow = await subscription.publish("token", {"content": "not-delivered"})
    after_detach = await subscription.publish("token", {"content": "also-not-delivered"})

    assert first is True
    assert overflow is False
    assert after_detach is False
    assert [event async for event in subscription.events()] == []
