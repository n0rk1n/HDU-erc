from __future__ import annotations

from uuid import uuid4

import aiosqlite

from chatbot.core.time import utc_now
from chatbot.db.connection import Database
from chatbot.db.models import Conversation


class ConversationRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def get_or_create_default(
        self, connection: aiosqlite.Connection, user_id: int
    ) -> Conversation:
        """Return the caller's existing default or create one inside its transaction."""
        conversation = await self._select_default(connection, user_id)
        if conversation is not None:
            return conversation

        now = utc_now()
        try:
            await connection.execute(
                """
                INSERT INTO conversations(
                    id, user_id, thread_id, is_default, title, status, created_at, updated_at
                ) VALUES (?, ?, ?, 1, '新对话', 'active', ?, ?)
                """,
                (str(uuid4()), user_id, str(uuid4()), now, now),
            )
        except aiosqlite.IntegrityError:
            conversation = await self._select_default(connection, user_id)
            if conversation is None:
                raise
            return conversation

        conversation = await self._select_default(connection, user_id)
        if conversation is None:  # Defensive: the insert must be observable in this transaction.
            raise RuntimeError("default conversation insert did not return a row")
        return conversation

    async def get_default_by_user(self, user_id: int) -> Conversation | None:
        """Find a default conversation by internal user ID only."""
        async with self.database.connect() as connection:
            return await self._select_default(connection, user_id)

    @staticmethod
    async def _select_default(
        connection: aiosqlite.Connection, user_id: int
    ) -> Conversation | None:
        cursor = await connection.execute(
            """
            SELECT id, user_id, thread_id, is_default, title, status, created_at, updated_at
            FROM conversations
            WHERE user_id = ? AND is_default = 1
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Conversation(
            id=row[0],
            user_id=row[1],
            thread_id=row[2],
            is_default=bool(row[3]),
            title=row[4],
            status=row[5],
            created_at=row[6],
            updated_at=row[7],
        )
