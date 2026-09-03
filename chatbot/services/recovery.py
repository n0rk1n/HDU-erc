from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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
    interrupted = await messages.fail_interrupted(error_code="process_interrupted")
    thread_ids = sorted({item.thread_id for item in interrupted})
    for thread_id in thread_ids:
        await checkpointer.adelete_thread(thread_id)
    return RecoveryReport(
        failed_messages=len(interrupted),
        reset_threads=len(thread_ids),
    )
