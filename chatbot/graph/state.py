from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict


class TurnState(TypedDict, total=False):
    user_id: int
    conversation_id: str
    request_id: str
    user_message_id: str
    assistant_message_id: str
    phase: str
    error_code: str | None


class EventPublisher(Protocol):
    async def publish(self, name: str, data: dict[str, object]) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class TurnContext:
    thread_id: str
    user_id: int
    conversation_id: str
    request_id: str
    user_message_id: str
    assistant_message_id: str
    publisher: EventPublisher
