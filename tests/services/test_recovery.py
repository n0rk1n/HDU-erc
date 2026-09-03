from __future__ import annotations

import pytest

from chatbot.db.models import InterruptedTurn
from chatbot.services.recovery import RecoveryReport, recover_interrupted_turns


@pytest.mark.asyncio
async def test_recovery_fails_interrupted_messages_and_deletes_each_thread_once() -> None:
    """Catches checkpoint cleanup being skipped or repeated per message."""
    interrupted = [
        InterruptedTurn("conversation-a", "thread-a", "message-1"),
        InterruptedTurn("conversation-a", "thread-a", "message-2"),
        InterruptedTurn("conversation-b", "thread-b", "message-3"),
    ]

    class Messages:
        async def fail_interrupted(self, *, error_code: str):
            assert error_code == "process_interrupted"
            return interrupted

    class Saver:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def adelete_thread(self, thread_id: str) -> None:
            self.deleted.append(thread_id)

    saver = Saver()
    report = await recover_interrupted_turns(Messages(), saver)

    assert report == RecoveryReport(failed_messages=3, reset_threads=2)
    assert saver.deleted == ["thread-a", "thread-b"]


@pytest.mark.asyncio
async def test_recovery_keeps_checkpoints_when_no_interrupted_message_exists() -> None:
    """Catches normal completed thread checkpoints being cleared at every startup."""
    class Messages:
        async def fail_interrupted(self, *, error_code: str):
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
        async def fail_interrupted(self, *, error_code: str):
            return [InterruptedTurn("conversation", "thread", "message")]

    class Saver:
        async def adelete_thread(self, thread_id: str) -> None:
            raise RuntimeError("saver unavailable")

    with pytest.raises(RuntimeError, match="saver unavailable"):
        await recover_interrupted_turns(Messages(), Saver())
