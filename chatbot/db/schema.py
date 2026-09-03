from __future__ import annotations

from chatbot.core.errors import ConfigError
from chatbot.db.connection import Database


SCHEMA_VERSION = 1


DDL_STATEMENTS = (
    """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        identifier TEXT COLLATE BINARY NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (length(identifier) BETWEEN 1 AND 128)
    )
    """,
    """
    CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        thread_id TEXT NOT NULL UNIQUE,
        is_default INTEGER NOT NULL DEFAULT 1,
        title TEXT NOT NULL DEFAULT '新对话',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        CHECK (is_default IN (0, 1)),
        CHECK (status IN ('active', 'archived'))
    )
    """,
    """
    CREATE UNIQUE INDEX ux_conversations_default_user
        ON conversations(user_id)
        WHERE is_default = 1
    """,
    """
    CREATE INDEX ix_conversations_user_updated
        ON conversations(user_id, updated_at DESC)
    """,
    """
    CREATE TABLE messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        sequence_no INTEGER NOT NULL,
        role TEXT NOT NULL,
        status TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        reasoning_content TEXT,
        trace_json TEXT NOT NULL DEFAULT '{"schema_version":1,"events":[]}',
        prompt_json TEXT,
        provider TEXT,
        model TEXT,
        parameters_json TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        total_tokens INTEGER,
        latency_ms INTEGER,
        finish_reason TEXT,
        error_code TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
        UNIQUE (conversation_id, sequence_no),
        UNIQUE (conversation_id, request_id, role),
        CHECK (role IN ('user', 'assistant')),
        CHECK (status IN ('pending', 'streaming', 'completed', 'failed')),
        CHECK (sequence_no > 0),
        CHECK (input_tokens IS NULL OR input_tokens >= 0),
        CHECK (output_tokens IS NULL OR output_tokens >= 0),
        CHECK (total_tokens IS NULL OR total_tokens >= 0),
        CHECK (latency_ms IS NULL OR latency_ms >= 0)
    )
    """,
    """
    CREATE INDEX ix_messages_conversation_sequence
        ON messages(conversation_id, sequence_no)
    """,
    """
    CREATE INDEX ix_messages_status_updated
        ON messages(status, updated_at)
    """,
)


async def initialize_schema(database: Database) -> None:
    async with database.connect() as connection:
        version = (await (await connection.execute("PRAGMA user_version")).fetchone())[0]
    if version > SCHEMA_VERSION:
        raise ConfigError("unsupported database schema version")
    if version == SCHEMA_VERSION:
        return

    async with database.transaction(immediate=True) as connection:
        version = (await (await connection.execute("PRAGMA user_version")).fetchone())[0]
        if version > SCHEMA_VERSION:
            raise ConfigError("unsupported database schema version")
        if version == SCHEMA_VERSION:
            return
        for statement in DDL_STATEMENTS:
            await connection.execute(statement)
        await connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
