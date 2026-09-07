from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from chatbot.db.messages import MessageRepository
from chatbot.emotion.graph import EmotionRuntime
from chatbot.emotion_gate.runtime import GateRuntime
from chatbot.llm.types import ChatModelAdapter


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class NodeDependencies:
    messages: MessageRepository
    model: ChatModelAdapter
    emotion: EmotionRuntime
    gate: GateRuntime
    context_message_limit: int = 40
    clock: Callable[[], datetime] = _utc_now
    flush_interval_seconds: float = 0.250
    flush_character_threshold: int = 256

    def __post_init__(self) -> None:
        if self.context_message_limit < 1:
            raise ValueError("context_message_limit must be positive")
        if self.flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        if self.flush_character_threshold < 1:
            raise ValueError("flush_character_threshold must be positive")
