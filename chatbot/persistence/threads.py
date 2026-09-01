"""Client-scoped thread directory backed by the LangGraph Store."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from time import time_ns
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

    async def list(
        self,
        client_id: str,
        *,
        exists: Callable[[str], Awaitable[bool]] | None = None,
    ) -> list[ThreadRecord]:
        items = []
        offset = 0
        while True:
            page = await self._store.asearch(
                self._namespace(client_id),
                limit=1_000,
                offset=offset,
            )
            items.extend(page)
            if len(page) < 1_000:
                break
            offset += len(page)
        records = [
            (
                self._from_value(item.value),
                self._record_sort_time(item.value),
                self._item_sort_time(item),
                str(item.key),
            )
            for item in items
        ]
        if exists is not None:
            retained = []
            for record, record_time, sort_time, item_key in records:
                if await exists(record.thread_id):
                    retained.append((record, record_time, sort_time, item_key))
                else:
                    await self.delete_record(client_id, record.thread_id)
            records = retained
        ordered = sorted(
            records,
            key=lambda item: (item[1], item[2], item[3]),
            reverse=True,
        )
        return [record for record, _, _, _ in ordered]

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
            "_sort_key": str(time_ns()),
        }

    @staticmethod
    def _item_sort_time(item: Any) -> int:
        """Use write time for legacy records and nanoseconds for current records."""
        explicit = item.value.get("_sort_key")
        if explicit is not None:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass
        timestamp = getattr(item, "updated_at", None) or getattr(
            item, "created_at", None
        )
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return int(timestamp.timestamp() * 1_000_000_000)
        return 0

    @staticmethod
    def _record_sort_time(value: dict[str, Any]) -> int:
        timestamp = datetime.fromisoformat(str(value["updated_at"]).replace("Z", "+00:00"))
        return int(timestamp.astimezone(timezone.utc).timestamp() * 1_000_000_000)

    @staticmethod
    def _from_value(value: dict[str, Any]) -> ThreadRecord:
        return ThreadRecord(
            thread_id=str(value["thread_id"]),
            title=str(value["title"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
        )
