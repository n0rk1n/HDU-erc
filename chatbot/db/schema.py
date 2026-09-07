from __future__ import annotations

import aiosqlite

from chatbot.core.errors import ConfigError
from chatbot.db.connection import Database


SCHEMA_VERSION = 3


V1_DDL_STATEMENTS = (
    """
    -- 用户表：保存可登录或发起对话的用户身份
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 用户自增主键
        identifier TEXT COLLATE BINARY NOT NULL UNIQUE,  -- 区分大小写的用户唯一标识
        created_at TEXT NOT NULL,  -- 用户创建时间，ISO 8601 格式
        updated_at TEXT NOT NULL,  -- 用户最后更新时间，ISO 8601 格式
        CHECK (length(identifier) BETWEEN 1 AND 128)
    )
    """,
    """
    -- 会话表：保存用户的独立对话线程及其生命周期状态
    CREATE TABLE conversations (
        id TEXT PRIMARY KEY,  -- 会话唯一标识
        user_id INTEGER NOT NULL,  -- 会话所属用户 ID
        thread_id TEXT NOT NULL UNIQUE,  -- LangGraph 使用的唯一线程标识
        is_default INTEGER NOT NULL DEFAULT 1,  -- 是否为用户默认会话：1 是，0 否
        title TEXT NOT NULL DEFAULT '新对话',  -- 会话标题
        status TEXT NOT NULL DEFAULT 'active',  -- 会话状态：active 或 archived
        created_at TEXT NOT NULL,  -- 会话创建时间，ISO 8601 格式
        updated_at TEXT NOT NULL,  -- 会话最后更新时间，ISO 8601 格式
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
    -- 消息表：保存会话消息、模型调用审计信息及处理结果
    CREATE TABLE messages (
        id TEXT PRIMARY KEY,  -- 消息唯一标识
        conversation_id TEXT NOT NULL,  -- 消息所属会话 ID
        request_id TEXT NOT NULL,  -- 产生该消息的请求唯一标识
        sequence_no INTEGER NOT NULL,  -- 消息在会话内的递增序号
        role TEXT NOT NULL,  -- 消息角色：user 或 assistant
        status TEXT NOT NULL,  -- 处理状态：pending、streaming、completed 或 failed
        content TEXT NOT NULL DEFAULT '',  -- 最终回复或用户输入正文
        reasoning_content TEXT,  -- 模型返回的推理内容
        trace_json TEXT NOT NULL DEFAULT '{"schema_version":1,"events":[]}',  -- 流式事件轨迹 JSON
        prompt_json TEXT,  -- 发送给模型的提示词快照 JSON
        provider TEXT,  -- 模型服务提供商
        model TEXT,  -- 模型名称
        parameters_json TEXT,  -- 模型调用参数快照 JSON
        input_tokens INTEGER,  -- 输入 token 数
        output_tokens INTEGER,  -- 输出 token 数
        total_tokens INTEGER,  -- 总 token 数
        latency_ms INTEGER,  -- 模型调用耗时，单位毫秒
        finish_reason TEXT,  -- 模型停止生成的原因
        error_code TEXT,  -- 处理失败时的错误码
        error_message TEXT,  -- 处理失败时的错误信息
        created_at TEXT NOT NULL,  -- 消息创建时间，ISO 8601 格式
        updated_at TEXT NOT NULL,  -- 消息最后更新时间，ISO 8601 格式
        completed_at TEXT,  -- 消息完成时间，ISO 8601 格式
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



V2_DDL_STATEMENTS = (
    """CREATE TABLE emotion_analyses (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        request_id TEXT NOT NULL,
        user_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
        status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
        snapshot_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT, raw_output TEXT, reasoning_content TEXT,
        response_metadata_json TEXT, error_json TEXT,
        input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens>=0),
        output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens>=0),
        total_tokens INTEGER CHECK(total_tokens IS NULL OR total_tokens>=0),
        latency_ms INTEGER CHECK(latency_ms IS NULL OR latency_ms>=0),
        finish_reason TEXT, created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
        UNIQUE(conversation_id,request_id)
    )""",
    "CREATE INDEX ix_emotion_status_created ON emotion_analyses(status,created_at)",
)
V3_DDL_STATEMENTS = (
    """CREATE TABLE emotion_gate_decisions (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        request_id TEXT NOT NULL,
        user_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
        status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
        action TEXT CHECK(action IN ('analyze','skip')),
        reason TEXT, snapshot_json TEXT NOT NULL DEFAULT '{}', error_json TEXT,
        analysis_id TEXT REFERENCES emotion_analyses(id),
        created_at TEXT NOT NULL, completed_at TEXT,
        UNIQUE(conversation_id,request_id)
    )""",
    "CREATE INDEX ix_gate_status_created ON emotion_gate_decisions(status,created_at)",
    """CREATE TABLE emotion_gate_attempts (
        id TEXT PRIMARY KEY,
        decision_id TEXT NOT NULL REFERENCES emotion_gate_decisions(id) ON DELETE CASCADE,
        attempt_no INTEGER NOT NULL CHECK(attempt_no>0),
        status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
        snapshot_json TEXT NOT NULL, facts_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL, completed_at TEXT,
        UNIQUE(decision_id,attempt_no)
    )""",
)
DDL_STATEMENTS = (*V1_DDL_STATEMENTS, *V2_DDL_STATEMENTS, *V3_DDL_STATEMENTS)

async def initialize_schema(database: Database) -> None:
    version = await _read_schema_version(database)
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
        migrations = {1: V1_DDL_STATEMENTS, 2: V2_DDL_STATEMENTS, 3: V3_DDL_STATEMENTS}
        for target in range(version + 1, SCHEMA_VERSION + 1):
            for statement in migrations[target]:
                await connection.execute(statement)
        await connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


async def _read_schema_version(database: Database) -> int:
    """Read the version without applying the application's persistent PRAGMAs."""
    database.path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(database.path) as connection:
        return (await (await connection.execute("PRAGMA user_version")).fetchone())[0]
