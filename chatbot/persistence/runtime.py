"""Lifecycle management for the official SQLite LangGraph adapters."""

from __future__ import annotations

import sqlite3
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from chatbot.core.config import GraphConfig


@dataclass(frozen=True)
class PersistenceHandles:
    checkpointer: AsyncSqliteSaver
    store: AsyncSqliteStore


class PersistenceOpenError(RuntimeError):
    """A configured persistence file is not a readable SQLite database."""


def _ensure_parent(database_path: str) -> None:
    Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def open_persistence(config: GraphConfig) -> AsyncIterator[PersistenceHandles]:
    """Open independent SQLite-backed checkpoint and Store adapters."""
    _ensure_parent(config.checkpoint_db_path)
    _ensure_parent(config.store_db_path)
    async with AsyncExitStack() as stack:
        checkpoint_conn = await stack.enter_async_context(aiosqlite.connect(config.checkpoint_db_path))
        checkpointer = AsyncSqliteSaver(
            checkpoint_conn,
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=None,
            ),
        )
        store = await stack.enter_async_context(AsyncSqliteStore.from_conn_string(config.store_db_path))
        try:
            await checkpointer.setup()
        except sqlite3.DatabaseError as exc:
            raise PersistenceOpenError(
                f"invalid checkpoint database: {config.checkpoint_db_path}"
            ) from exc
        try:
            await store.setup()
        except sqlite3.DatabaseError as exc:
            raise PersistenceOpenError(
                f"invalid store database: {config.store_db_path}"
            ) from exc
        yield PersistenceHandles(checkpointer=checkpointer, store=store)
