from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import aiosqlite

from chatbot.core.errors import InvalidMessageState
from chatbot.core.time import utc_now
from chatbot.db.connection import Database
from chatbot.db.models import InterruptedTurn, Message, ReservedTurn


_MESSAGE_COLUMNS = """
    id, conversation_id, request_id, sequence_no, role, status, content,
    reasoning_content, trace_json, prompt_json, provider, model, parameters_json,
    input_tokens, output_tokens, total_tokens, latency_ms, finish_reason,
    error_code, error_message, created_at, updated_at, completed_at
"""
_UNSET = object()


class MessageRepository:
    """Persist complete message facts independently from graph checkpoints."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def reserve_turn(
        self, conversation_id: str, request_id: str, content: str
    ) -> ReservedTurn:
        async with self.database.transaction(immediate=True) as connection:
            existing = await self._select_turn(connection, conversation_id, request_id)
            if existing is not None:
                return existing

            next_sequence = (
                await (
                    await connection.execute(
                        "SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM messages WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                ).fetchone()
            )[0]
            now = utc_now()
            user_id = str(uuid4())
            assistant_id = str(uuid4())
            await connection.executemany(
                """
                INSERT INTO messages(
                    id, conversation_id, request_id, sequence_no, role, status,
                    content, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        user_id,
                        conversation_id,
                        request_id,
                        next_sequence,
                        "user",
                        "completed",
                        content,
                        now,
                        now,
                        now,
                    ),
                    (
                        assistant_id,
                        conversation_id,
                        request_id,
                        next_sequence + 1,
                        "assistant",
                        "pending",
                        "",
                        now,
                        now,
                        None,
                    ),
                ),
            )
            turn = await self._select_turn(connection, conversation_id, request_id)
            if turn is None:
                raise RuntimeError("reserved message pair was not readable")
            return turn

    async def find_turn(
        self, conversation_id: str, request_id: str
    ) -> ReservedTurn | None:
        async with self.database.connect() as connection:
            return await self._select_turn(connection, conversation_id, request_id)

    async def find_assistant(
        self, conversation_id: str, request_id: str
    ) -> Message | None:
        async with self.database.connect() as connection:
            cursor = await connection.execute(
                f"""
                SELECT {_MESSAGE_COLUMNS}
                FROM messages
                WHERE conversation_id = ? AND request_id = ? AND role = 'assistant'
                """,
                (conversation_id, request_id),
            )
            row = await cursor.fetchone()
            return None if row is None else _message_from_row(row)

    async def list_visible(
        self,
        conversation_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[dict[str, object | None]]:
        """Return only fields approved for ordinary history responses."""
        condition = "conversation_id = ?"
        parameters: list[object] = [conversation_id]
        if before_sequence is not None:
            condition += " AND sequence_no < ?"
            parameters.append(before_sequence)
        parameters.append(limit)
        async with self.database.connect() as connection:
            cursor = await connection.execute(
                f"""
                SELECT id, request_id, sequence_no, role, status, content,
                       error_code, error_message, created_at, updated_at, completed_at
                FROM messages
                WHERE {condition}
                ORDER BY sequence_no DESC
                LIMIT ?
                """,
                tuple(parameters),
            )
            rows = await cursor.fetchall()
        keys = (
            "id",
            "request_id",
            "sequence_no",
            "role",
            "status",
            "content",
            "error_code",
            "error_message",
            "created_at",
            "updated_at",
            "completed_at",
        )
        return [dict(zip(keys, row, strict=True)) for row in reversed(rows)]

    async def list_context(
        self, conversation_id: str, *, limit: int
    ) -> list[dict[str, str]]:
        """Load the recent completed role/content facts in conversational order."""
        async with self.database.connect() as connection:
            cursor = await connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ? AND status = 'completed'
                ORDER BY sequence_no DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = await cursor.fetchall()
        return [dict(zip(("role", "content"), row, strict=True)) for row in reversed(rows)]

    async def mark_streaming(self, assistant_message_id: str) -> Message:
        now = utc_now()
        async with self.database.transaction(immediate=True) as connection:
            cursor = await connection.execute(
                """
                UPDATE messages
                SET status = 'streaming', updated_at = ?
                WHERE id = ? AND role = 'assistant' AND status = 'pending'
                """,
                (now, assistant_message_id),
            )
            _require_single_update(cursor)
            return await self._select_message_by_id(connection, assistant_message_id)

    async def flush_partial(
        self,
        assistant_message_id: str,
        *,
        content: str,
        reasoning_content: str | None,
        trace: object,
    ) -> Message:
        now = utc_now()
        async with self.database.transaction(immediate=True) as connection:
            cursor = await connection.execute(
                """
                UPDATE messages
                SET content = ?, reasoning_content = ?, trace_json = ?, updated_at = ?
                WHERE id = ? AND role = 'assistant' AND status = 'streaming'
                """,
                (
                    content,
                    reasoning_content,
                    _stable_json(trace),
                    now,
                    assistant_message_id,
                ),
            )
            _require_single_update(cursor)
            return await self._select_message_by_id(connection, assistant_message_id)

    async def complete_assistant(
        self,
        assistant_message_id: str,
        *,
        content: str,
        reasoning_content: str | None,
        trace: object,
        prompt: object,
        provider: str | None,
        model: str | None,
        parameters: object | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        latency_ms: int | None,
        finish_reason: str | None,
    ) -> Message:
        """Store supplied provider facts; never synthesize hidden reasoning."""
        now = utc_now()
        async with self.database.transaction(immediate=True) as connection:
            cursor = await connection.execute(
                """
                UPDATE messages
                SET status = 'completed', content = ?, reasoning_content = ?,
                    trace_json = ?, prompt_json = ?, provider = ?, model = ?,
                    parameters_json = ?, input_tokens = ?, output_tokens = ?,
                    total_tokens = ?, latency_ms = ?, finish_reason = ?,
                    error_code = NULL, error_message = NULL,
                    updated_at = ?, completed_at = ?
                WHERE id = ? AND role = 'assistant' AND status = 'streaming'
                """,
                (
                    content,
                    reasoning_content,
                    _stable_json(trace),
                    _stable_json(prompt),
                    provider,
                    model,
                    None if parameters is None else _stable_json(parameters),
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    latency_ms,
                    finish_reason,
                    now,
                    now,
                    assistant_message_id,
                ),
            )
            _require_single_update(cursor)
            return await self._select_message_by_id(connection, assistant_message_id)

    async def fail_assistant(
        self,
        assistant_message_id: str,
        *,
        error_code: str,
        error_message: str,
        content: str | object = _UNSET,
        reasoning_content: str | None | object = _UNSET,
        trace: object = _UNSET,
    ) -> Message:
        """Fail an active assistant while retaining any partial facts not supplied here."""
        assignments = ["status = 'failed'", "error_code = ?", "error_message = ?"]
        values: list[object | None] = [error_code, error_message]
        if content is not _UNSET:
            assignments.append("content = ?")
            values.append(content)
        if reasoning_content is not _UNSET:
            assignments.append("reasoning_content = ?")
            values.append(reasoning_content)
        if trace is not _UNSET:
            assignments.append("trace_json = ?")
            values.append(_stable_json(trace))
        now = utc_now()
        assignments.extend(("updated_at = ?", "completed_at = ?"))
        values.extend((now, now, assistant_message_id))
        async with self.database.transaction(immediate=True) as connection:
            cursor = await connection.execute(
                f"""
                UPDATE messages
                SET {', '.join(assignments)}
                WHERE id = ? AND role = 'assistant' AND status = 'streaming'
                """,
                tuple(values),
            )
            _require_single_update(cursor)
            return await self._select_message_by_id(connection, assistant_message_id)

    async def fail_interrupted(
        self,
        *,
        error_code: str,
        error_message: str = "generation interrupted",
        stale_before: str | None = None,
    ) -> list[InterruptedTurn]:
        cutoff = utc_now() if stale_before is None else stale_before
        now = utc_now()
        async with self.database.transaction(immediate=True) as connection:
            cursor = await connection.execute(
                """
                SELECT m.id, m.conversation_id, c.thread_id
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE m.role = 'assistant'
                  AND m.status IN ('pending', 'streaming')
                  AND m.updated_at < ?
                ORDER BY m.conversation_id, m.sequence_no
                """,
                (cutoff,),
            )
            rows = await cursor.fetchall()
            interrupted: list[InterruptedTurn] = []
            seen: set[tuple[str, str, str]] = set()
            for message_id, conversation_id, thread_id in rows:
                update = await connection.execute(
                    """
                    UPDATE messages
                    SET status = 'failed', error_code = ?, error_message = ?,
                        updated_at = ?, completed_at = ?
                    WHERE id = ? AND role = 'assistant'
                      AND status IN ('pending', 'streaming') AND updated_at < ?
                    """,
                    (error_code, error_message, now, now, message_id, cutoff),
                )
                _require_single_update(update)
                key = (conversation_id, thread_id, message_id)
                if key not in seen:
                    seen.add(key)
                    interrupted.append(InterruptedTurn(*key))
            return interrupted

    @staticmethod
    async def _select_turn(
        connection: aiosqlite.Connection, conversation_id: str, request_id: str
    ) -> ReservedTurn | None:
        cursor = await connection.execute(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM messages
            WHERE conversation_id = ? AND request_id = ?
            ORDER BY sequence_no
            """,
            (conversation_id, request_id),
        )
        rows = await cursor.fetchall()
        if not rows:
            return None
        messages = [_message_from_row(row) for row in rows]
        by_role = {message.role: message for message in messages}
        if len(messages) != 2 or set(by_role) != {"user", "assistant"}:
            raise InvalidMessageState("request does not contain one complete message pair")
        return ReservedTurn(user=by_role["user"], assistant=by_role["assistant"])

    @staticmethod
    async def _select_message_by_id(
        connection: aiosqlite.Connection, message_id: str
    ) -> Message:
        cursor = await connection.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE id = ?", (message_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise InvalidMessageState()
        return _message_from_row(row)


def _require_single_update(cursor: aiosqlite.Cursor) -> None:
    if cursor.rowcount != 1:
        raise InvalidMessageState()


def _message_from_row(row: Sequence[Any]) -> Message:
    return Message(*row)


def _stable_json(value: object) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
