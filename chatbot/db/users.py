from __future__ import annotations

import aiosqlite

from chatbot.core.time import utc_now
from chatbot.db.models import User


class UserRepository:
    async def get_or_create(self, connection: aiosqlite.Connection, identifier: str) -> User:
        """Create an exact-match user or return the existing user within the caller's transaction."""
        now = utc_now()
        cursor = await connection.execute(
            """
            INSERT INTO users(identifier, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(identifier) DO UPDATE SET updated_at = users.updated_at
            RETURNING id, identifier, created_at, updated_at
            """,
            (identifier, now, now),
        )
        row = await cursor.fetchone()
        if row is None:  # SQLite's RETURNING contract makes this defensive only.
            raise RuntimeError("user insert did not return a row")
        return User(id=row[0], identifier=row[1], created_at=row[2], updated_at=row[3])
