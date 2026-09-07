"""SQLite 业务表及增量建表语句。

SQLite 没有独立的列 COMMENT 元数据；列备注以 SQL 注释保存在新建表的
sqlite_schema.sql 中。仅修改这些注释不会更新已经创建的表，因此不提升版本号。
"""

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
    """
    -- 情绪分析表：保存每轮实际触发的情绪识别结果及模型调用审计
    CREATE TABLE emotion_analyses (
        id TEXT PRIMARY KEY,  -- 情绪分析记录唯一标识
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,  -- 所属会话 ID，随会话级联删除
        request_id TEXT NOT NULL,  -- 触发本次分析的请求唯一标识
        user_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,  -- 触发分析的用户消息 ID，每条消息最多一条分析记录
        status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),  -- 分析状态：待处理、运行中、已完成或失败
        snapshot_json TEXT NOT NULL DEFAULT '{}',  -- 分析请求快照 JSON，包含提示词、模型参数、标签、示例及上下文预算信息
        result_json TEXT,  -- 解析并校验后的结构化情绪识别结果 JSON
        raw_output TEXT,  -- 情绪模型返回的原始输出正文
        reasoning_content TEXT,  -- 供应商实际返回的推理内容，未返回时为空
        response_metadata_json TEXT,  -- 模型响应元数据 JSON
        error_json TEXT,  -- 分析失败诊断 JSON，包含失败阶段、错误码及错误信息
        input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens>=0),  -- 模型输入 token 数，未获取时为空
        output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens>=0),  -- 模型输出 token 数，未获取时为空
        total_tokens INTEGER CHECK(total_tokens IS NULL OR total_tokens>=0),  -- 模型调用总 token 数，未获取时为空
        latency_ms INTEGER CHECK(latency_ms IS NULL OR latency_ms>=0),  -- 模型调用耗时，单位毫秒，未获取时为空
        finish_reason TEXT,  -- 模型停止生成的原因
        created_at TEXT NOT NULL,  -- 分析记录创建时间，UTC ISO 8601 格式
        started_at TEXT,  -- 分析开始时间，UTC ISO 8601 格式，未开始时为空
        completed_at TEXT,  -- 分析结束时间，UTC ISO 8601 格式，完成或失败时记录
        UNIQUE(conversation_id,request_id)
    )
    """,
    "CREATE INDEX ix_emotion_status_created ON emotion_analyses(status,created_at)",
)
V3_DDL_STATEMENTS = (
    """
    -- 情绪入口判定表：保存每轮是否执行情绪分析的决定及判定依据
    CREATE TABLE emotion_gate_decisions (
        id TEXT PRIMARY KEY,  -- 入口判定记录唯一标识
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,  -- 所属会话 ID，随会话级联删除
        request_id TEXT NOT NULL,  -- 触发本次判定的请求唯一标识
        user_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,  -- 触发判定的用户消息 ID，每条消息最多一条判定记录
        status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),  -- 判定状态：待处理、运行中、已完成或失败
        action TEXT CHECK(action IN ('analyze','skip')),  -- 最终动作：analyze 执行情绪分析，skip 跳过；未决定时为空
        reason TEXT,  -- 判定原因或触发规则标识
        snapshot_json TEXT NOT NULL DEFAULT '{}',  -- 判定快照 JSON，包含配置、轮次及历史情绪来源等依据
        error_json TEXT,  -- 判定错误诊断 JSON，失败降级后也可保留错误信息
        analysis_id TEXT REFERENCES emotion_analyses(id),  -- 本次判定触发的情绪分析记录 ID，未关联时为空
        created_at TEXT NOT NULL,  -- 判定记录创建时间，UTC ISO 8601 格式
        completed_at TEXT,  -- 判定结束时间，UTC ISO 8601 格式，完成或失败时记录
        UNIQUE(conversation_id,request_id)
    )
    """,
    "CREATE INDEX ix_gate_status_created ON emotion_gate_decisions(status,created_at)",
    """
    -- 情绪判定调用表：逐次保存判定 Agent 的模型调用与重试审计
    CREATE TABLE emotion_gate_attempts (
        id TEXT PRIMARY KEY,  -- 判定模型单次调用记录唯一标识
        decision_id TEXT NOT NULL REFERENCES emotion_gate_decisions(id) ON DELETE CASCADE,  -- 所属入口判定 ID，随判定记录级联删除
        attempt_no INTEGER NOT NULL CHECK(attempt_no>0),  -- 同一判定内的调用序号，从 1 开始递增
        status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),  -- 调用状态：运行中、已完成或失败
        snapshot_json TEXT NOT NULL,  -- 本次调用的输入快照 JSON，包含提示词、模型配置及上下文
        facts_json TEXT NOT NULL DEFAULT '{}',  -- 本次调用的结果事实 JSON，包含输出、token 用量、耗时及错误等审计信息
        created_at TEXT NOT NULL,  -- 本次调用开始记录时间，UTC ISO 8601 格式
        completed_at TEXT,  -- 本次调用结束时间，UTC ISO 8601 格式，完成或失败时记录
        UNIQUE(decision_id,attempt_no)
    )
    """,
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
