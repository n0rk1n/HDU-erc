"""Client-scoped long-term memory backed by LangGraph's Store protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from langgraph.store.base import BaseStore

from chatbot.memory.models import MEMORY_CATEGORIES, Memory, MemoryCandidate
from chatbot.memory.rules import (
    _tokens,
    can_supersede,
    conflicts,
    deterministic_memory_key,
    is_similar_memory,
    normalize_content,
    ranking_score,
)


class StoreMemoryRepository:
    """Persist client-scoped long-term memories in a supplied LangGraph Store."""

    def __init__(self, store: BaseStore, *, now: Callable[[], datetime] | None = None) -> None:
        self._store = store
        self._now = now or (lambda: datetime.now(timezone.utc))

    async def asearch(self, client_id: str, query: str, *, limit: int) -> list[Memory]:
        query_tokens = _tokens(query)
        if not query_tokens or limit <= 0:
            return []
        memories = await self._active_memories(client_id)
        scored = []
        for memory in memories:
            score = ranking_score(memory, query, query_tokens)
            if score > 0:
                scored.append((score, memory.updated_at, memory.use_count, memory))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected = [item[3] for item in scored[:limit]]
        if not selected:
            return []
        now = self._timestamp()
        result = []
        for memory in selected:
            item = await self._store.aget(self._memory_namespace(client_id), memory.id)
            if item is None or item.value.get("status") != "active":
                continue
            value = {**item.value, "last_used_at": now, "use_count": memory.use_count + 1}
            await self._store.aput(self._memory_namespace(client_id), memory.id, value)
            result.append(self._memory_from_value(memory.id, value))
        return result

    async def aremember(
        self, client_id: str, candidates: Iterable[MemoryCandidate]
    ) -> list[Memory]:
        stored: list[Memory] = []
        for candidate in candidates:
            content = normalize_content(candidate.content)
            if not content or candidate.category not in MEMORY_CATEGORIES:
                continue
            active = await self._active_memories(client_id)
            existing = next(
                (memory for memory in active if is_similar_memory(memory, candidate, content)), None
            )
            if existing is not None:
                stored.append(await self._update(client_id, existing, candidate, content))
                continue
            conflicting = [
                memory for memory in active if conflicts(memory, candidate, content)
            ]
            if any(not can_supersede(memory, candidate) for memory in conflicting):
                continue
            for memory in conflicting:
                await self._set_status(client_id, memory.id, "superseded")
            stored.append(
                await self._insert(
                    client_id,
                    candidate,
                    content,
                    supersedes_id=conflicting[0].id if conflicting else None,
                )
            )
        return stored

    async def aget_consolidation_state(self, client_id: str) -> dict[str, Any]:
        item = await self._store.aget(self._meta_namespace(client_id), "consolidation")
        if item is None:
            return self._default_consolidation_state()
        value = item.value
        checkpoints = value.get("processed_checkpoint_ids", [])
        return {
            "last_turn_count": int(value.get("last_turn_count", 0)),
            "last_message_id": value.get("last_message_id"),
            "processed_checkpoint_ids": [str(item) for item in checkpoints if item],
        }

    async def amark_consolidated(
        self,
        client_id: str,
        *,
        turn_count: int,
        last_message_id: str | None,
        source_checkpoint_id: str | None,
    ) -> None:
        state = await self.aget_consolidation_state(client_id)
        checkpoints = state["processed_checkpoint_ids"]
        if source_checkpoint_id and source_checkpoint_id not in checkpoints:
            checkpoints = [*checkpoints, source_checkpoint_id]
        await self._store.aput(
            self._meta_namespace(client_id),
            "consolidation",
            {
                "last_turn_count": turn_count,
                "last_message_id": last_message_id,
                "processed_checkpoint_ids": checkpoints,
                "updated_at": self._timestamp(),
            },
        )

    async def _active_memories(self, client_id: str) -> list[Memory]:
        items = await self._store.asearch(self._memory_namespace(client_id), limit=1_000)
        return [
            self._memory_from_value(item.key, item.value)
            for item in items
            if item.value.get("status") == "active"
        ]

    async def _insert(
        self,
        client_id: str,
        candidate: MemoryCandidate,
        content: str,
        *,
        supersedes_id: str | None,
    ) -> Memory:
        memory_id = deterministic_memory_key(content)
        now = self._timestamp()
        value = {
            "id": memory_id,
            "content": content,
            "category": candidate.category,
            "source": candidate.source,
            "confidence": candidate.confidence,
            "created_at": now,
            "updated_at": now,
            "last_used_at": None,
            "use_count": 0,
            "status": "active",
            "supersedes_id": supersedes_id,
        }
        await self._store.aput(self._memory_namespace(client_id), memory_id, value)
        return self._memory_from_value(memory_id, value)

    async def _update(
        self,
        client_id: str,
        existing: Memory,
        candidate: MemoryCandidate,
        content: str,
    ) -> Memory:
        item = await self._store.aget(self._memory_namespace(client_id), existing.id)
        if item is None:
            return await self._insert(client_id, candidate, content, supersedes_id=None)
        value = {
            **item.value,
            "content": content,
            "category": candidate.category,
            "source": candidate.source,
            "confidence": max(existing.confidence, candidate.confidence),
            "updated_at": self._timestamp(),
        }
        await self._store.aput(self._memory_namespace(client_id), existing.id, value)
        return self._memory_from_value(existing.id, value)

    async def _set_status(self, client_id: str, memory_id: str, status: str) -> None:
        item = await self._store.aget(self._memory_namespace(client_id), memory_id)
        if item is not None:
            await self._store.aput(
                self._memory_namespace(client_id), memory_id, {**item.value, "status": status}
            )

    def _timestamp(self) -> str:
        now = self._now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Memory timestamps must be timezone-aware UTC datetimes.")
        return now.astimezone(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _memory_namespace(client_id: str) -> tuple[str, str]:
        return client_id, "memories"

    @staticmethod
    def _meta_namespace(client_id: str) -> tuple[str, str]:
        return client_id, "memory_meta"

    @staticmethod
    def _memory_from_value(memory_id: str, value: dict[str, Any]) -> Memory:
        return Memory(
            id=str(value.get("id", memory_id)),
            content=str(value["content"]),
            category=str(value["category"]),
            source=str(value["source"]),
            confidence=float(value["confidence"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            last_used_at=value.get("last_used_at"),
            use_count=int(value.get("use_count", 0)),
        )

    @staticmethod
    def _default_consolidation_state() -> dict[str, Any]:
        return {
            "last_turn_count": 0,
            "last_message_id": None,
            "processed_checkpoint_ids": [],
        }
