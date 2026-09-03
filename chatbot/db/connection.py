from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.path)
        try:
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute("PRAGMA synchronous = NORMAL")
            yield connection
        finally:
            await connection.close()

    @asynccontextmanager
    async def transaction(self, immediate: bool = False) -> AsyncIterator[aiosqlite.Connection]:
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()
