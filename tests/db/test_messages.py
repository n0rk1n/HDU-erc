from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
import pytest_asyncio

import chatbot.db.messages as messages_module
from chatbot.core.errors import InvalidMessageState
from chatbot.db.messages import MessageRepository
from chatbot.services.identity import IdentityService


REQUEST_ID = "f5317709-b72b-4b58-a7f8-b5cc49a568ee"


@pytest_asyncio.fixture
async def conversation(database):
    return (await IdentityService(database).resolve("message-owner")).conversation


@pytest_asyncio.fixture
async def messages(database) -> MessageRepository:
    return MessageRepository(database)


@pytest_asyncio.fixture
async def inject_message_facts(database, conversation):
    async def inject(request_id: str, facts: list[tuple[str, int]]) -> None:
        timestamp = "2026-09-03T00:00:00+00:00"
        async with database.transaction(immediate=True) as connection:
            for index, (role, sequence_no) in enumerate(facts):
                status = "completed" if role == "user" else "pending"
                await connection.execute(
                    """
                    INSERT INTO messages(
                        id, conversation_id, request_id, sequence_no, role, status,
                        content, created_at, updated_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"malformed-{request_id}-{index}",
                        conversation.id,
                        request_id,
                        sequence_no,
                        role,
                        status,
                        role,
                        timestamp,
                        timestamp,
                        timestamp if role == "user" else None,
                    ),
                )

    return inject


@pytest.mark.asyncio
async def test_reserve_turn_writes_pair_with_stable_sequence(messages, conversation):
    """Catches separate inserts or sequence allocation that loses the adjacent turn pair."""
    turn = await messages.reserve_turn(conversation.id, REQUEST_ID, "你好")

    assert turn.user.sequence_no == 1
    assert turn.user.status == "completed"
    assert turn.user.content == "你好"
    assert turn.user.completed_at is not None
    assert turn.assistant.sequence_no == 2
    assert turn.assistant.status == "pending"
    assert turn.user.request_id == turn.assistant.request_id == REQUEST_ID


@pytest.mark.asyncio
async def test_reserve_same_request_returns_existing_pair_without_overwriting_content(
    messages, conversation
):
    """Catches retries that duplicate a turn or rewrite the original user fact."""
    first = await messages.reserve_turn(conversation.id, REQUEST_ID, "你好")
    second = await messages.reserve_turn(conversation.id, REQUEST_ID, "不同内容")

    assert second == first
    assert second.user.content == "你好"


@pytest.mark.asyncio
async def test_concurrent_reservations_allocate_unique_adjacent_sequences(messages, conversation):
    """Catches MAX(sequence_no) races that duplicate or interleave turn pairs."""
    turns = await asyncio.gather(
        *(
            messages.reserve_turn(conversation.id, str(uuid4()), f"message-{index}")
            for index in range(8)
        )
    )

    pairs = sorted((turn.user.sequence_no, turn.assistant.sequence_no) for turn in turns)
    assert pairs == [(index, index + 1) for index in range(1, 16, 2)]


@pytest.mark.asyncio
async def test_find_turn_and_assistant_return_persisted_facts(messages, conversation):
    """Catches request lookup that cannot reliably replay the original pair."""
    reserved = await messages.reserve_turn(conversation.id, REQUEST_ID, "你好")

    assert await messages.find_turn(conversation.id, REQUEST_ID) == reserved
    assert await messages.find_assistant(conversation.id, REQUEST_ID) == reserved.assistant
    assert await messages.find_turn(conversation.id, "missing") is None
    assert await messages.find_assistant(conversation.id, "missing") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "facts",
    (
        [("assistant", 1)],
        [("user", 1)],
        [("user", 1), ("assistant", 3)],
        [("assistant", 1), ("user", 2)],
    ),
    ids=("orphan-assistant", "missing-assistant", "non-adjacent", "roles-reversed"),
)
async def test_find_queries_reject_incomplete_or_misordered_request_facts(
    messages, conversation, inject_message_facts, facts
):
    """Catches replay lookups treating corrupt or half-written request facts as a valid turn."""
    request_id = "malformed-request"
    await inject_message_facts(request_id, facts)

    with pytest.raises(InvalidMessageState) as turn_error:
        await messages.find_turn(conversation.id, request_id)
    with pytest.raises(InvalidMessageState) as assistant_error:
        await messages.find_assistant(conversation.id, request_id)

    assert turn_error.value.code == "invalid_message"
    assert assistant_error.value.code == "invalid_message"


@pytest.mark.asyncio
async def test_list_visible_pages_latest_rows_in_display_order_without_private_fields(
    messages, conversation
):
    """Catches reversed pagination or ordinary history leaking private audit material."""
    for index in range(3):
        await messages.reserve_turn(conversation.id, f"request-{index}", f"user-{index}")

    page = await messages.list_visible(conversation.id, limit=3)
    older = await messages.list_visible(
        conversation.id, limit=2, before_sequence=page[0]["sequence_no"]
    )

    assert [item["sequence_no"] for item in page] == [4, 5, 6]
    assert [item["sequence_no"] for item in older] == [2, 3]
    assert set(page[0]) == {
        "id",
        "request_id",
        "sequence_no",
        "role",
        "status",
        "content",
        "error_code",
        "error_message",
        "created_at",
        "updated_at",
        "completed_at",
    }
    assert not {
        "conversation_id",
        "thread_id",
        "reasoning_content",
        "trace_json",
        "prompt_json",
        "parameters_json",
    } & set(page[0])


@pytest.mark.asyncio
async def test_list_context_keeps_only_recent_completed_messages_in_conversation_order(
    messages, conversation
):
    """Catches incomplete or failed assistant output entering the next model prompt."""
    first = await messages.reserve_turn(conversation.id, "request-1", "user-1")
    await messages.mark_streaming(first.assistant.id)
    await messages.complete_assistant(
        first.assistant.id,
        content="assistant-1",
        reasoning_content="provider reasoning",
        trace={"schema_version": 1, "nodes": []},
        prompt=[{"role": "user", "content": "user-1"}],
        provider="openai-compatible",
        model="test-model",
        parameters={"temperature": 0.2},
        input_tokens=4,
        output_tokens=2,
        total_tokens=6,
        latency_ms=12,
        finish_reason="stop",
    )
    failed = await messages.reserve_turn(conversation.id, "request-2", "user-2")
    await messages.mark_streaming(failed.assistant.id)
    await messages.flush_partial(
        failed.assistant.id,
        content="partial-secret",
        reasoning_content="actual provider reasoning",
        trace={"schema_version": 1, "events": [{"name": "chunk"}]},
    )
    await messages.fail_assistant(
        failed.assistant.id,
        error_code="model_error",
        error_message="safe failure",
    )
    pending = await messages.reserve_turn(conversation.id, "request-3", "user-3")

    context = await messages.list_context(conversation.id, limit=3)
    stored_failed = await messages.find_assistant(conversation.id, "request-2")

    assert context == [
        {"role": "assistant", "content": "assistant-1"},
        {"role": "user", "content": "user-2"},
        {"role": "user", "content": "user-3"},
    ]
    assert stored_failed is not None
    assert stored_failed.content == "partial-secret"
    assert stored_failed.reasoning_content == "actual provider reasoning"
    assert pending.assistant.status == "pending"


@pytest.mark.asyncio
async def test_flush_partial_updates_content_reasoning_trace_and_timestamp(
    messages, conversation, monkeypatch
):
    """Catches partial durability that updates only visible text or leaves stale audit time."""
    turn = await messages.reserve_turn(conversation.id, REQUEST_ID, "你好")
    await messages.mark_streaming(turn.assistant.id)
    before = (await messages.find_assistant(conversation.id, REQUEST_ID)).updated_at
    monkeypatch.setattr(messages_module, "utc_now", lambda: "2026-09-03T12:34:56+00:00")

    updated = await messages.flush_partial(
        turn.assistant.id,
        content="部分回答",
        reasoning_content="供应商实际返回的 reasoning",
        trace={"z": 1, "schema_version": 1, "events": []},
    )

    assert updated.content == "部分回答"
    assert updated.reasoning_content == "供应商实际返回的 reasoning"
    assert updated.trace_json == '{"events":[],"schema_version":1,"z":1}'
    assert updated.updated_at == "2026-09-03T12:34:56+00:00"
    assert updated.updated_at != before


@pytest.mark.asyncio
async def test_complete_assistant_atomically_persists_all_audit_fields(
    messages, conversation, monkeypatch
):
    """Catches successful completion that drops model inputs, usage, timing, or finish metadata."""
    turn = await messages.reserve_turn(conversation.id, REQUEST_ID, "你好")
    await messages.mark_streaming(turn.assistant.id)
    monkeypatch.setattr(messages_module, "utc_now", lambda: "2026-09-03T13:00:00+00:00")

    completed = await messages.complete_assistant(
        turn.assistant.id,
        content="最终回答",
        reasoning_content="实际 reasoning",
        trace={"schema_version": 1, "model_calls": [{"call_id": "call-1"}]},
        prompt=[{"content": "你好", "role": "user"}],
        provider="openai-compatible",
        model="deepseek-chat",
        parameters={"temperature": 0.1, "max_tokens": 256},
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        latency_ms=456,
        finish_reason="stop",
    )

    assert completed.status == "completed"
    assert completed.content == "最终回答"
    assert completed.reasoning_content == "实际 reasoning"
    assert json.loads(completed.trace_json)["model_calls"][0]["call_id"] == "call-1"
    assert completed.prompt_json == '[{"content":"你好","role":"user"}]'
    assert completed.provider == "openai-compatible"
    assert completed.model == "deepseek-chat"
    assert completed.parameters_json == '{"max_tokens":256,"temperature":0.1}'
    assert (completed.input_tokens, completed.output_tokens, completed.total_tokens) == (10, 20, 30)
    assert completed.latency_ms == 456
    assert completed.finish_reason == "stop"
    assert completed.updated_at == completed.completed_at == "2026-09-03T13:00:00+00:00"


@pytest.mark.asyncio
async def test_success_updates_conversation_once_but_failure_and_replay_do_not(
    messages, conversation, database, monkeypatch
) -> None:
    """Catches completion ordering time drifting on failures or terminal replay."""
    async def conversation_updated_at() -> str:
        async with database.connect() as connection:
            row = await (
                await connection.execute(
                    "SELECT updated_at FROM conversations WHERE id = ?", (conversation.id,)
                )
            ).fetchone()
        assert row is not None
        return row[0]

    success = await messages.reserve_turn(conversation.id, "success-touch", "one")
    await messages.mark_streaming(success.assistant.id)
    monkeypatch.setattr(messages_module, "utc_now", lambda: "2026-09-03T14:00:00+00:00")
    completed = await messages.complete_assistant(
        success.assistant.id,
        content="done",
        reasoning_content=None,
        trace={"schema_version": 1},
        prompt=[],
        provider="test",
        model="test",
        parameters={},
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        latency_ms=1,
        finish_reason="stop",
    )
    assert await conversation_updated_at() == completed.updated_at

    failed = await messages.reserve_turn(conversation.id, "failure-no-touch", "two")
    await messages.mark_streaming(failed.assistant.id)
    monkeypatch.setattr(messages_module, "utc_now", lambda: "2026-09-03T15:00:00+00:00")
    await messages.fail_assistant(
        failed.assistant.id, error_code="model_error", error_message="safe"
    )
    assert await conversation_updated_at() == "2026-09-03T14:00:00+00:00"

    monkeypatch.setattr(messages_module, "utc_now", lambda: "2026-09-03T16:00:00+00:00")
    with pytest.raises(InvalidMessageState):
        await messages.complete_assistant(
            success.assistant.id,
            content="replay",
            reasoning_content=None,
            trace={"schema_version": 1},
            prompt=[],
            provider="test",
            model="test",
            parameters={},
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=1,
            finish_reason="stop",
        )
    assert await conversation_updated_at() == "2026-09-03T14:00:00+00:00"


@pytest.mark.asyncio
async def test_completion_rolls_back_when_conversation_timestamp_cannot_update(
    messages, conversation, database, monkeypatch
) -> None:
    """Catches assistant completion committing without its conversation ordering write."""
    turn = await messages.reserve_turn(conversation.id, "conversation-rollback", "one")
    await messages.mark_streaming(turn.assistant.id)
    async with database.transaction(immediate=True) as connection:
        await connection.execute(
            """
            CREATE TRIGGER reject_conversation_touch
            BEFORE UPDATE OF updated_at ON conversations
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """
        )
    monkeypatch.setattr(messages_module, "utc_now", lambda: "2026-09-03T17:00:00+00:00")

    with pytest.raises(InvalidMessageState):
        await messages.complete_assistant(
            turn.assistant.id,
            content="must-roll-back",
            reasoning_content=None,
            trace={"schema_version": 1},
            prompt=[],
            provider="test",
            model="test",
            parameters={},
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=1,
            finish_reason="stop",
        )

    assistant = await messages.find_assistant(conversation.id, "conversation-rollback")
    assert assistant is not None
    assert assistant.status == "streaming"
    assert assistant.content == ""


@pytest.mark.asyncio
async def test_lifecycle_rejects_skips_repeated_updates_and_terminal_overwrites(
    messages, conversation
):
    """Catches state updates that skip pending->streaming or overwrite terminal facts."""
    pending = await messages.reserve_turn(conversation.id, "request-pending", "one")
    with pytest.raises(InvalidMessageState) as skipped:
        await messages.complete_assistant(
            pending.assistant.id,
            content="invalid",
            reasoning_content=None,
            trace={"schema_version": 1},
            prompt=[],
            provider=None,
            model=None,
            parameters=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            latency_ms=None,
            finish_reason=None,
        )
    assert skipped.value.code == "invalid_message"

    await messages.mark_streaming(pending.assistant.id)
    with pytest.raises(InvalidMessageState):
        await messages.mark_streaming(pending.assistant.id)
    await messages.fail_assistant(
        pending.assistant.id, error_code="model_error", error_message="failed"
    )
    with pytest.raises(InvalidMessageState):
        await messages.flush_partial(
            pending.assistant.id,
            content="overwrite",
            reasoning_content="overwrite",
            trace={"schema_version": 1},
        )
    with pytest.raises(InvalidMessageState):
        await messages.fail_assistant(
            pending.assistant.id, error_code="other", error_message="overwrite"
        )

    failed = await messages.find_assistant(conversation.id, "request-pending")
    assert failed.status == "failed"
    assert failed.error_code == "model_error"
    assert failed.error_message == "failed"


@pytest.mark.asyncio
async def test_list_interrupted_is_read_only_and_uses_the_recovery_cutoff(
    messages, conversation, database
):
    """Catches recovery discovery mutating rows before checkpoint cleanup succeeds."""
    stale = await messages.reserve_turn(conversation.id, "request-stale-list", "one")
    fresh = await messages.reserve_turn(conversation.id, "request-fresh-list", "two")
    async with database.transaction(immediate=True) as connection:
        await connection.execute(
            "UPDATE messages SET updated_at = ? WHERE id = ?",
            ("2026-09-02T00:00:00+00:00", stale.assistant.id),
        )
        await connection.execute(
            "UPDATE messages SET updated_at = ? WHERE id = ?",
            ("2026-09-03T12:00:00+00:00", fresh.assistant.id),
        )

    interrupted = await messages.list_interrupted(
        stale_before="2026-09-03T00:00:00+00:00"
    )

    assert interrupted == [
        type(interrupted[0])(
            conversation_id=conversation.id,
            thread_id=conversation.thread_id,
            assistant_message_id=stale.assistant.id,
        )
    ]
    assert (await messages.find_assistant(conversation.id, "request-stale-list")).status == "pending"
    assert (await messages.find_assistant(conversation.id, "request-fresh-list")).status == "pending"


@pytest.mark.asyncio
async def test_fail_interrupted_marks_only_stale_active_assistants_and_preserves_partials(
    messages, conversation, database
):
    """Catches recovery failing fresh/terminal rows or erasing partial generation evidence."""
    stale_pending = await messages.reserve_turn(conversation.id, "request-stale-pending", "one")
    stale_streaming = await messages.reserve_turn(conversation.id, "request-stale-streaming", "two")
    await messages.mark_streaming(stale_streaming.assistant.id)
    await messages.flush_partial(
        stale_streaming.assistant.id,
        content="partial",
        reasoning_content="actual reasoning",
        trace={"schema_version": 1, "events": [{"name": "token"}]},
    )
    terminal = await messages.reserve_turn(conversation.id, "request-terminal", "three")
    await messages.mark_streaming(terminal.assistant.id)
    await messages.fail_assistant(
        terminal.assistant.id, error_code="model_error", error_message="already failed"
    )
    fresh = await messages.reserve_turn(conversation.id, "request-fresh", "four")

    old_time = "2026-09-02T00:00:00+00:00"
    fresh_time = "2026-09-03T12:00:00+00:00"
    async with database.transaction(immediate=True) as connection:
        await connection.execute(
            "UPDATE messages SET updated_at = ? WHERE id IN (?, ?)",
            (old_time, stale_pending.assistant.id, stale_streaming.assistant.id),
        )
        await connection.execute(
            "UPDATE messages SET updated_at = ? WHERE id = ?", (fresh_time, fresh.assistant.id)
        )

    interrupted = await messages.fail_interrupted(
        error_code="process_interrupted",
        error_message="generation interrupted by restart",
        stale_before="2026-09-03T00:00:00+00:00",
    )

    assert interrupted == [
        type(interrupted[0])(
            conversation_id=conversation.id,
            thread_id=conversation.thread_id,
            assistant_message_id=stale_pending.assistant.id,
        ),
        type(interrupted[0])(
            conversation_id=conversation.id,
            thread_id=conversation.thread_id,
            assistant_message_id=stale_streaming.assistant.id,
        ),
    ]
    recovered = await messages.find_assistant(conversation.id, "request-stale-streaming")
    assert recovered.status == "failed"
    assert recovered.content == "partial"
    assert recovered.reasoning_content == "actual reasoning"
    assert json.loads(recovered.trace_json)["events"] == [{"name": "token"}]
    assert recovered.error_code == "process_interrupted"
    assert (await messages.find_assistant(conversation.id, "request-terminal")).error_code == "model_error"
    assert (await messages.find_assistant(conversation.id, "request-fresh")).status == "pending"
