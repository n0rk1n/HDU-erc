from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chatbot.core.time import utc_now
from chatbot.db.messages import MessageRepository


class ThreadCheckpointer(Protocol):
    async def adelete_thread(self, thread_id: str) -> None: ...


@dataclass(frozen=True)
class RecoveryReport:
    failed_messages: int
    reset_threads: int


async def recover_interrupted_turns(
    messages: MessageRepository,
    checkpointer: AsyncSqliteSaver | ThreadCheckpointer,
) -> RecoveryReport:
    cutoff = utc_now()
    interrupted = await messages.list_interrupted(stale_before=cutoff)
    thread_ids = sorted({item.thread_id for item in interrupted})
    for thread_id in thread_ids:
        await checkpointer.adelete_thread(thread_id)
    failed = await messages.fail_interrupted(
        error_code="process_interrupted",
        stale_before=cutoff,
    )
    return RecoveryReport(
        failed_messages=len(failed),
        reset_threads=len(thread_ids),
    )
