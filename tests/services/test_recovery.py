from __future__ import annotations

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chatbot.db.messages import MessageRepository
from chatbot.db.models import InterruptedTurn
from chatbot.services.recovery import RecoveryReport, recover_interrupted_turns
from chatbot.services.identity import IdentityService


@pytest.mark.asyncio
async def test_recovery_fails_interrupted_messages_and_deletes_each_thread_once() -> None:
    """Catches checkpoint cleanup being skipped or repeated per message."""
    interrupted = [
        InterruptedTurn("conversation-a", "thread-a", "message-1"),
        InterruptedTurn("conversation-a", "thread-a", "message-2"),
        InterruptedTurn("conversation-b", "thread-b", "message-3"),
    ]
    actions: list[str] = []

    class Messages:
        async def list_interrupted(self, *, stale_before: str):
            actions.append("list")
            return interrupted

        async def fail_interrupted(self, *, error_code: str, stale_before: str):
            assert error_code == "process_interrupted"
            actions.append("fail")
            return interrupted

    class Saver:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def adelete_thread(self, thread_id: str) -> None:
            actions.append(f"delete:{thread_id}")
            self.deleted.append(thread_id)

    saver = Saver()
    report = await recover_interrupted_turns(Messages(), saver)

    assert report == RecoveryReport(failed_messages=3, reset_threads=2)
    assert saver.deleted == ["thread-a", "thread-b"]
    assert actions == ["list", "delete:thread-a", "delete:thread-b", "fail"]


@pytest.mark.asyncio
async def test_recovery_keeps_checkpoints_when_no_interrupted_message_exists() -> None:
    """Catches normal completed thread checkpoints being cleared at every startup."""
    class Messages:
        async def list_interrupted(self, *, stale_before: str):
            return []

        async def fail_interrupted(self, *, error_code: str, stale_before: str):
            return []

    class Saver:
        async def adelete_thread(self, thread_id: str) -> None:
            raise AssertionError(f"normal checkpoint deleted: {thread_id}")

    report = await recover_interrupted_turns(Messages(), Saver())

    assert report == RecoveryReport(failed_messages=0, reset_threads=0)


@pytest.mark.asyncio
async def test_recovery_delete_failure_aborts_startup_contract() -> None:
    """Catches partial checkpoint-reset failures being hidden behind a healthy report."""
    class Messages:
        fail_calls = 0

        async def list_interrupted(self, *, stale_before: str):
            return [InterruptedTurn("conversation", "thread", "message")]

        async def fail_interrupted(self, *, error_code: str, stale_before: str):
            self.fail_calls += 1
            return [InterruptedTurn("conversation", "thread", "message")]

    class Saver:
        async def adelete_thread(self, thread_id: str) -> None:
            raise RuntimeError("saver unavailable")

    messages = Messages()
    with pytest.raises(RuntimeError, match="saver unavailable"):
        await recover_interrupted_turns(messages, Saver())
    assert messages.fail_calls == 0


async def _put_checkpoint(checkpointer: AsyncSqliteSaver, thread_id: str) -> None:
    await checkpointer.aput(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        empty_checkpoint(),
        {"source": "input", "step": -1, "parents": {}},
        {},
    )


@pytest.mark.asyncio
async def test_checkpoint_delete_failure_keeps_message_retryable_until_next_recovery(
    database,
) -> None:
    """Catches a failed delete making its still-present checkpoint undiscoverable forever."""
    identity = await IdentityService(database).resolve("retry-delete")
    messages = MessageRepository(database)
    request_id = "00000000-0000-4000-8000-000000000001"
    await messages.reserve_turn(identity.conversation.id, request_id, "hello")

    async with AsyncSqliteSaver.from_conn_string(str(database.path)) as saver:
        await saver.setup()
        await _put_checkpoint(saver, identity.conversation.thread_id)

        class FailFirstDelete:
            def __init__(self) -> None:
                self.calls = 0

            async def adelete_thread(self, thread_id: str) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("checkpoint delete unavailable")
                await saver.adelete_thread(thread_id)

        flaky_saver = FailFirstDelete()
        with pytest.raises(RuntimeError, match="checkpoint delete unavailable"):
            await recover_interrupted_turns(messages, flaky_saver)

        after_failure = await messages.find_assistant(
            identity.conversation.id, request_id
        )
        checkpoint_after_failure = await saver.aget_tuple(
            {"configurable": {"thread_id": identity.conversation.thread_id}}
        )

        report = await recover_interrupted_turns(messages, flaky_saver)
        after_retry = await messages.find_assistant(identity.conversation.id, request_id)
        checkpoint_after_retry = await saver.aget_tuple(
            {"configurable": {"thread_id": identity.conversation.thread_id}}
        )

    assert after_failure is not None
    assert after_failure.status == "pending"
    assert checkpoint_after_failure is not None
    assert report == RecoveryReport(failed_messages=1, reset_threads=1)
    assert flaky_saver.calls == 2
    assert after_retry is not None
    assert after_retry.status == "failed"
    assert after_retry.error_code == "process_interrupted"
    assert checkpoint_after_retry is None


@pytest.mark.asyncio
async def test_business_failure_after_delete_is_retryable_with_idempotent_redelete(
    database,
) -> None:
    """Catches checkpoint success plus business failure leaving an unrecoverable active turn."""
    identity = await IdentityService(database).resolve("retry-business")
    real_messages = MessageRepository(database)
    request_id = "00000000-0000-4000-8000-000000000002"
    await real_messages.reserve_turn(identity.conversation.id, request_id, "hello")

    class FailFirstBusinessUpdate:
        def __init__(self) -> None:
            self.fail_calls = 0

        async def list_interrupted(self, *, stale_before: str):
            return await real_messages.list_interrupted(stale_before=stale_before)

        async def fail_interrupted(self, **kwargs):
            self.fail_calls += 1
            if self.fail_calls == 1:
                raise RuntimeError("business database unavailable")
            return await real_messages.fail_interrupted(**kwargs)

    messages = FailFirstBusinessUpdate()
    async with AsyncSqliteSaver.from_conn_string(str(database.path)) as saver:
        await saver.setup()
        await _put_checkpoint(saver, identity.conversation.thread_id)

        with pytest.raises(RuntimeError, match="business database unavailable"):
            await recover_interrupted_turns(messages, saver)

        after_failure = await real_messages.find_assistant(
            identity.conversation.id, request_id
        )
        checkpoint_after_failure = await saver.aget_tuple(
            {"configurable": {"thread_id": identity.conversation.thread_id}}
        )
        report = await recover_interrupted_turns(messages, saver)
        after_retry = await real_messages.find_assistant(
            identity.conversation.id, request_id
        )

    assert after_failure is not None
    assert after_failure.status == "pending"
    assert checkpoint_after_failure is None
    assert report == RecoveryReport(failed_messages=1, reset_threads=1)
    assert after_retry is not None
    assert after_retry.status == "failed"
