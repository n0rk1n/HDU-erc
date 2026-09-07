from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import aiosqlite
from aiosqlite import IntegrityError

from chatbot.core.errors import ConfigError
from chatbot.db.connection import Database
from chatbot.db.models import Conversation, InterruptedTurn, Message, ReservedTurn, User
from chatbot.db.schema import initialize_schema, SCHEMA_VERSION


async def execute(connection, sql: str, parameters: tuple = ()) -> None:
    await connection.execute(sql, parameters)
    await connection.commit()


@pytest.mark.asyncio
async def test_schema_creates_three_business_tables_and_version(tmp_path) -> None:
    """Catches a deployment that has no durable business schema or schema version."""
    database = Database(tmp_path / "chatbot.sqlite3")
    await initialize_schema(database)

    async with database.connect() as connection:
        names = {
            row[0]
            for row in await (
                await connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).fetchall()
        }
        version = (await (await connection.execute("PRAGMA user_version")).fetchone())[0]

    assert {"users", "conversations", "messages"} <= names
    assert "checkpoints" not in names
    assert version == SCHEMA_VERSION


@pytest.mark.asyncio
async def test_database_connections_apply_required_pragmas(database: Database) -> None:
    """Catches connections that would bypass WAL, foreign keys, timeout, or safe sync mode."""
    async with database.connect() as connection:
        journal_mode = (await (await connection.execute("PRAGMA journal_mode")).fetchone())[0]
        foreign_keys = (await (await connection.execute("PRAGMA foreign_keys")).fetchone())[0]
        busy_timeout = (await (await connection.execute("PRAGMA busy_timeout")).fetchone())[0]
        synchronous = (await (await connection.execute("PRAGMA synchronous")).fetchone())[0]

    assert journal_mode == "wal"
    assert foreign_keys == 1
    assert busy_timeout == 5000
    assert synchronous == 1


@pytest.mark.asyncio
async def test_transaction_commits_success_and_rolls_back_error(database: Database) -> None:
    """Catches transactions that leak partial writes after a repository error."""
    async with database.transaction(immediate=True) as connection:
        await connection.execute("CREATE TABLE transaction_probe (value TEXT NOT NULL)")
        await connection.execute("INSERT INTO transaction_probe(value) VALUES ('committed')")

    with pytest.raises(RuntimeError, match="rollback"):
        async with database.transaction(immediate=True) as connection:
            await connection.execute("INSERT INTO transaction_probe(value) VALUES ('rolled-back')")
            raise RuntimeError("rollback")

    async with database.connect() as connection:
        values = [
            row[0]
            for row in await (
                await connection.execute("SELECT value FROM transaction_probe ORDER BY rowid")
            ).fetchall()
        ]
    assert values == ["committed"]


@pytest.mark.asyncio
async def test_schema_enforces_identifier_and_default_conversation_rules(database: Database) -> None:
    """Catches missing BINARY identity, identifier checks, or partial default uniqueness."""
    async with database.connect() as connection:
        await execute(
            connection,
            "INSERT INTO users(identifier, created_at, updated_at) VALUES (?, ?, ?)",
            ("Alice", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
        )
        await execute(
            connection,
            "INSERT INTO users(identifier, created_at, updated_at) VALUES (?, ?, ?)",
            ("alice", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
        )
        with pytest.raises(IntegrityError):
            await execute(
                connection,
                "INSERT INTO users(identifier, created_at, updated_at) VALUES (?, ?, ?)",
                ("", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
            )
        await execute(
            connection,
            "INSERT INTO conversations(id, user_id, thread_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("conversation-1", 1, "thread-1", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
        )
        with pytest.raises(IntegrityError):
            await execute(
                connection,
                "INSERT INTO conversations(id, user_id, thread_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("conversation-2", 1, "thread-2", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
            )
        await execute(
            connection,
            "INSERT INTO conversations(id, user_id, thread_id, is_default, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
            ("conversation-3", 1, "thread-3", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
        )
        with pytest.raises(IntegrityError):
            await execute(
                connection,
                "INSERT INTO conversations(id, user_id, thread_id, is_default, created_at, updated_at) VALUES (?, ?, ?, 2, ?, ?)",
                ("conversation-4", 1, "thread-4", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
            )


@pytest.mark.asyncio
async def test_schema_enforces_message_lifecycle_and_audit_constraints(database: Database) -> None:
    """Catches invalid lifecycle records, duplicate request roles, or negative telemetry."""
    async with database.connect() as connection:
        await execute(
            connection,
            "INSERT INTO users(identifier, created_at, updated_at) VALUES (?, ?, ?)",
            ("Alice", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
        )
        await execute(
            connection,
            "INSERT INTO conversations(id, user_id, thread_id, created_at, updated_at) VALUES (?, 1, ?, ?, ?)",
            ("conversation-1", "thread-1", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
        )
        await execute(
            connection,
            "INSERT INTO messages(id, conversation_id, request_id, sequence_no, role, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("message-1", "conversation-1", "request-1", 1, "user", "completed", "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
        )
        default_trace = (await (await connection.execute("SELECT trace_json FROM messages WHERE id = 'message-1'")).fetchone())[0]
        assert default_trace == '{"schema_version":1,"events":[]}'
        for values in [
            ("message-2", "request-1", 2, "user", "completed", None),
            ("message-3", "request-2", 1, "assistant", "pending", None),
            ("message-4", "request-3", 3, "tool", "completed", None),
            ("message-5", "request-4", 4, "assistant", "unknown", None),
            ("message-6", "request-5", 0, "assistant", "pending", None),
            ("message-7", "request-6", 5, "assistant", "pending", -1),
        ]:
            message_id, request_id, sequence_no, role, status, input_tokens = values
            with pytest.raises(IntegrityError):
                await execute(
                    connection,
                    "INSERT INTO messages(id, conversation_id, request_id, sequence_no, role, status, input_tokens, created_at, updated_at) VALUES (?, 'conversation-1', ?, ?, ?, ?, ?, ?, ?)",
                    (message_id, request_id, sequence_no, role, status, input_tokens, "2026-09-03T00:00:00+00:00", "2026-09-03T00:00:00+00:00"),
                )


@pytest.mark.asyncio
async def test_schema_is_idempotent_and_rejects_future_versions(tmp_path) -> None:
    """Catches accidental repeated DDL and opening a database the app cannot safely understand."""
    database = Database(tmp_path / "chatbot.sqlite3")
    await initialize_schema(database)
    await initialize_schema(database)
    async with database.connect() as connection:
        await connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(ConfigError, match="unsupported database schema version"):
        await initialize_schema(database)


@pytest.mark.asyncio
async def test_schema_rejects_future_version_without_mutating_database(tmp_path) -> None:
    """Catches version preflight changing an unsupported database before rejecting it."""
    path = tmp_path / "future-schema.sqlite3"
    async with aiosqlite.connect(path) as connection:
        await connection.execute("PRAGMA journal_mode = DELETE")
        await connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        await connection.commit()

    with pytest.raises(ConfigError, match="unsupported database schema version"):
        await initialize_schema(Database(path))

    async with aiosqlite.connect(path) as connection:
        version = (await (await connection.execute("PRAGMA user_version")).fetchone())[0]
        journal_mode = (await (await connection.execute("PRAGMA journal_mode")).fetchone())[0]
        tables = {
            row[0]
            for row in await (
                await connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).fetchall()
        }

    assert version == SCHEMA_VERSION + 1
    assert journal_mode == "delete"
    assert not {"users", "conversations", "messages"} & tables


def test_domain_models_are_immutable_and_retain_interrupted_turn_identifiers() -> None:
    """Catches mutable persistence facts or recovery records that omit graph routing identifiers."""
    user = User(id=1, identifier="Alice", created_at="2026-09-03T00:00:00+00:00", updated_at="2026-09-03T00:00:00+00:00")
    conversation = Conversation(
        id="conversation-1", user_id=1, thread_id="thread-1", is_default=True,
        title="新对话", status="active", created_at="2026-09-03T00:00:00+00:00", updated_at="2026-09-03T00:00:00+00:00",
    )
    message = Message(
        id="message-1", conversation_id="conversation-1", request_id="request-1", sequence_no=1,
        role="user", status="completed", content="你好", reasoning_content=None,
        trace_json='{"schema_version":1,"events":[]}', prompt_json=None, provider=None, model=None,
        parameters_json=None, input_tokens=None, output_tokens=None, total_tokens=None, latency_ms=None,
        finish_reason=None, error_code=None, error_message=None, created_at="2026-09-03T00:00:00+00:00",
        updated_at="2026-09-03T00:00:00+00:00", completed_at="2026-09-03T00:00:00+00:00",
    )
    turn = ReservedTurn(user=message, assistant=message)
    interrupted = InterruptedTurn(conversation_id="conversation-1", thread_id="thread-1", assistant_message_id="message-2")

    assert turn.user.id == "message-1"
    assert interrupted == InterruptedTurn("conversation-1", "thread-1", "message-2")
    with pytest.raises(FrozenInstanceError):
        user.identifier = "Bob"  # type: ignore[misc]
