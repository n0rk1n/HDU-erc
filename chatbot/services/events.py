from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, cast


EventName = Literal["run_started", "user_message", "token", "bubble", "done", "error", "progress"]
_EVENT_NAMES = frozenset({"run_started", "user_message", "token", "bubble", "done", "error", "progress"})
_END = object()


@dataclass(frozen=True)
class SseEvent:
    name: EventName
    data: dict[str, object]


class TurnSubscription:
    """A bounded, detachable display channel independent from turn production.

    The payload limit is ``queue_capacity``. One additional queue slot is reserved
    for the private end marker. A slow consumer that fills the payload allowance is
    detached: queued display events are discarded and production remains nonblocking.
    """

    def __init__(self, *, queue_capacity: int = 64) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self._payload_capacity = queue_capacity
        self._queue: asyncio.Queue[SseEvent | object] = asyncio.Queue(
            maxsize=queue_capacity + 1
        )
        self._detached = False
        self._closed = False

    async def publish(self, name: str, data: dict[str, object]) -> bool:
        """Publish without awaiting consumer capacity; overflow detaches the display."""
        if name not in _EVENT_NAMES:
            raise ValueError("unsupported SSE event")
        if self._detached or self._closed:
            return False
        if self._queue.qsize() >= self._payload_capacity:
            self.detach()
            return False
        self._queue.put_nowait(SseEvent(cast(EventName, name), dict(data)))
        return True

    def close(self) -> None:
        if self._detached or self._closed:
            return
        self._closed = True
        self._queue.put_nowait(_END)

    def detach(self) -> None:
        if self._detached:
            return
        self._detached = True
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(_END)

    async def events(self) -> AsyncIterator[SseEvent]:
        try:
            while True:
                item = await self._queue.get()
                if item is _END:
                    return
                yield cast(SseEvent, item)
        finally:
            self.detach()
