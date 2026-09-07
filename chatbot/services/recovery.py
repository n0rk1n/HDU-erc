from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chatbot.core.time import utc_now
from chatbot.db.messages import MessageRepository
from chatbot.db.emotions import EmotionRepository
from chatbot.db.emotion_gates import GateRepository


class ThreadCheckpointer(Protocol):
    async def adelete_thread(self, thread_id: str) -> None: ...


@dataclass(frozen=True)
class RecoveryReport:
    failed_messages: int
    reset_threads: int
    failed_analyses: int = 0
    failed_gate_decisions: int = 0


async def recover_interrupted_turns(
    messages: MessageRepository,
    checkpointer: AsyncSqliteSaver | ThreadCheckpointer,
    *, emotions: EmotionRepository | None = None, gates: GateRepository | None = None,
) -> RecoveryReport:
    cutoff = utc_now()
    failed_gates = await (gates or GateRepository(messages.database)).fail_interrupted(cutoff=cutoff)
    failed_analyses = await (emotions or EmotionRepository(messages.database)).fail_interrupted(cutoff=cutoff)
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
        failed_analyses=failed_analyses,
        failed_gate_decisions=failed_gates,
    )
