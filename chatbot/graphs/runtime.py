"""Application-lifetime facade around the persistent conversation graph."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import StateSnapshot

from chatbot.core.config import ChatConfig, GraphConfig
from chatbot.graphs.conversation import build_conversation_graph
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.memory import StoreMemoryRepository
from chatbot.models.graph import GraphContext, ProfileAnswer, ThreadRecord
from chatbot.persistence.runtime import PersistenceHandles
from chatbot.persistence.threads import ThreadRepository


class RuntimeOperationError(ValueError):
    """Stable facade failure suitable for mapping at the API boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ConversationRuntime:
    """Own one compiled graph and serialize mutations within each thread."""

    def __init__(
        self,
        graph: Any,
        thread_repository: ThreadRepository,
        checkpointer: Any,
        *,
        dependencies: NodeDependencies | None = None,
    ) -> None:
        self.graph = graph
        self.compiled_graph = graph
        self.thread_repository = thread_repository
        self.checkpointer = checkpointer
        self.dependencies = dependencies
        self.memory_repository = (
            dependencies.memory_repository if dependencies is not None else None
        )
        self._thread_locks: dict[str, asyncio.Lock] = {}

    async def acreate_thread(
        self,
        client_id: str,
        *,
        title: str = "新对话",
    ) -> ThreadRecord:
        """Create a directory record and its initial checkpoint atomically enough to roll back."""
        record = await self.thread_repository.create(client_id, title=title)
        try:
            await self.graph.aupdate_state(
                self._config(record.thread_id),
                {
                    "thread_meta": {
                        "thread_id": record.thread_id,
                        "title": record.title,
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                    }
                },
            )
        except Exception:
            await self.thread_repository.delete_record(client_id, record.thread_id)
            raise
        return record

    async def astream_turn(
        self,
        client_id: str,
        thread_id: str,
        request_id: str,
        message: str,
    ) -> AsyncIterator[Any]:
        """Stream one complete turn while retaining the thread lock through cancellation."""
        async for part in self._astream_locked(
            client_id,
            thread_id,
            request_id,
            {
                "operation": "turn",
                "request_id": request_id,
                "input_message": message,
            },
        ):
            yield part

    async def astream_regeneration(
        self,
        client_id: str,
        thread_id: str,
        request_id: str,
        message_id: str,
        reason: str,
    ) -> AsyncIterator[Any]:
        """Stream one same-ID regeneration under the thread mutation lock."""
        async for part in self._astream_locked(
            client_id,
            thread_id,
            request_id,
            {
                "operation": "regenerate",
                "request_id": request_id,
                "target_message_id": message_id,
                "regeneration_reason": reason,
            },
        ):
            yield part

    async def ainvoke_profile_draft(
        self,
        client_id: str,
        thread_id: str,
        request_id: str,
        answers: list[ProfileAnswer],
    ) -> dict[str, str]:
        """Produce an unpersisted profile proposal for an owned thread."""
        await self._require_thread(client_id, thread_id)
        result = await self.graph.ainvoke(
            {
                "operation": "onboard",
                "request_id": request_id,
                "profile_answers": answers,
            },
            self._config(thread_id),
            context=self._context(client_id, request_id),
        )
        return dict(result.get("profile_draft", {}))

    async def aget_state(self, client_id: str, thread_id: str) -> StateSnapshot:
        """Return the current official StateSnapshot for an owned thread."""
        await self._require_thread(client_id, thread_id)
        return await self.graph.aget_state(self._config(thread_id))

    async def aupdate_message_feedback(
        self,
        client_id: str,
        thread_id: str,
        message_id: str,
        feedback: str,
    ) -> AIMessage:
        """Replace one unrated AIMessage under the same ID and preserve all other fields."""
        lock = self.lock_for(thread_id)
        await lock.acquire()
        try:
            await self._require_thread(client_id, thread_id)
            if feedback not in {"like", "dislike"}:
                raise RuntimeOperationError("invalid_feedback")
            snapshot = await self.graph.aget_state(self._config(thread_id))
            target = next(
                (
                    message
                    for message in snapshot.values.get("messages", [])
                    if getattr(message, "id", None) == message_id
                ),
                None,
            )
            if not isinstance(target, AIMessage):
                raise RuntimeOperationError("message_not_found")
            if target.additional_kwargs.get("feedback") in {"like", "dislike"}:
                raise RuntimeOperationError("already_rated")
            replacement = target.model_copy(
                update={
                    "additional_kwargs": {
                        **target.additional_kwargs,
                        "feedback": feedback,
                    }
                }
            )
            await self.graph.aupdate_state(
                self._config(thread_id),
                {"messages": [replacement]},
                as_node="turn",
            )
            return replacement
        finally:
            lock.release()

    async def adelete_thread(self, client_id: str, thread_id: str) -> None:
        """Delete Saver state before its directory record, preserving recovery semantics."""
        lock = self.lock_for(thread_id)
        await lock.acquire()
        try:
            await self._require_thread(client_id, thread_id)
            try:
                await self.checkpointer.adelete_thread(thread_id)
            except Exception as exc:
                raise RuntimeOperationError("thread_delete_failed") from exc
            try:
                await self.thread_repository.delete_record(client_id, thread_id)
            except Exception as exc:
                raise RuntimeOperationError("thread_delete_failed") from exc
        finally:
            lock.release()

    def lock_for(self, thread_id: str) -> asyncio.Lock:
        """Return the stable application-lifetime lock for one thread."""
        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_locks[thread_id] = lock
        return lock

    async def _astream_locked(
        self,
        client_id: str,
        thread_id: str,
        request_id: str,
        graph_input: dict[str, Any],
    ) -> AsyncIterator[Any]:
        lock = self.lock_for(thread_id)
        await lock.acquire()
        try:
            await self._require_thread(client_id, thread_id)
            async for part in self.graph.astream(
                graph_input,
                self._config(thread_id),
                context=self._context(client_id, request_id),
                stream_mode=["messages", "custom"],
                subgraphs=True,
                version="v2",
            ):
                yield part
            await self.thread_repository.touch(client_id, thread_id)
        finally:
            lock.release()

    async def _require_thread(self, client_id: str, thread_id: str) -> None:
        if not await self.thread_repository.owns(client_id, thread_id):
            raise RuntimeOperationError("thread_not_found")

    async def _checkpoint_exists(self, thread_id: str) -> bool:
        return await self.checkpointer.aget_tuple(self._config(thread_id)) is not None

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _context(client_id: str, request_id: str) -> GraphContext:
        return GraphContext(client_id=client_id, request_id=request_id, locale="zh-CN")


def build_graph_runtime(
    handles: PersistenceHandles,
    chat_config: ChatConfig,
    graph_config: GraphConfig,
    *,
    model_factory: Callable[[ChatConfig], tuple[Any, Any]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> ConversationRuntime:
    """Build models, repositories, dependencies, and the parent graph exactly once."""
    if model_factory is None:
        from chatbot.main import build_runtime_llms

        model_factory = build_runtime_llms
    clock = now or _utc_now
    chat_model, emotion_model = model_factory(chat_config)
    memory_repository = StoreMemoryRepository(handles.store, now=clock)
    dependencies = NodeDependencies(
        chat_model=chat_model,
        emotion_model=emotion_model,
        chat_config=chat_config,
        graph_config=graph_config,
        memory_repository=memory_repository,
        now=clock,
    )
    compiled_graph = build_conversation_graph(dependencies).compile(
        checkpointer=handles.checkpointer,
        store=handles.store,
    )
    thread_repository = ThreadRepository(handles.store, now=clock)
    return ConversationRuntime(
        compiled_graph,
        thread_repository,
        handles.checkpointer,
        dependencies=dependencies,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
