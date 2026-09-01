"""Client-scoped thread directory backed by the LangGraph Store."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langgraph.store.base import BaseStore

from chatbot.models.graph import ThreadRecord


class ThreadRepository:
    """Persist thread metadata independently from checkpointed graph state."""

    def __init__(self, store: BaseStore, *, now: Callable[[], datetime] | None = None) -> None:
        self._store = store
        self._now = now or (lambda: datetime.now(timezone.utc))

    async def create(self, client_id: str, *, title: str = "新对话") -> ThreadRecord:
        timestamp = self._timestamp()
        record = ThreadRecord(
            thread_id=str(uuid4()),
            title=title,
            created_at=timestamp,
            updated_at=timestamp,
        )
        await self._store.aput(self._namespace(client_id), record.thread_id, self._as_value(record))
        return record

    async def list(self, client_id: str) -> list[ThreadRecord]:
        items = await self._store.asearch(self._namespace(client_id), limit=1_000)
        records = [self._from_value(item.value) for item in items]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    async def owns(self, client_id: str, thread_id: str) -> bool:
        return await self._store.aget(self._namespace(client_id), thread_id) is not None

    async def touch(self, client_id: str, thread_id: str, *, title: str | None = None) -> None:
        item = await self._store.aget(self._namespace(client_id), thread_id)
        if item is None:
            return
        current = self._from_value(item.value)
        record = ThreadRecord(
            thread_id=current.thread_id,
            title=current.title if title is None else title,
            created_at=current.created_at,
            updated_at=self._timestamp(),
        )
        await self._store.aput(self._namespace(client_id), thread_id, self._as_value(record))

    async def delete_record(self, client_id: str, thread_id: str) -> None:
        await self._store.adelete(self._namespace(client_id), thread_id)

    @staticmethod
    def _namespace(client_id: str) -> tuple[str, str]:
        return client_id, "threads"

    def _timestamp(self) -> str:
        now = self._now()
        if now.tzinfo is None:
            raise ValueError("Thread timestamps must be timezone-aware UTC datetimes.")
        return now.astimezone(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _as_value(record: ThreadRecord) -> dict[str, str]:
        return {
            "thread_id": record.thread_id,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _from_value(value: dict[str, Any]) -> ThreadRecord:
        return ThreadRecord(
            thread_id=str(value["thread_id"]),
            title=str(value["title"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
        )
