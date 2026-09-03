# LangGraph Multi-User ChatBot Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有情绪陪伴项目完整归档，并交付一个使用 LangGraph、FastAPI、SSE 和单文件 SQLite 的多用户 ChatBot 首期版本。

**Architecture:** 三个业务表 `users`、`conversations`、`messages` 保存可查询事实，官方 `AsyncSqliteSaver` 在同一 SQLite 文件中保存轻量图状态。FastAPI 负责用户解析和 SSE，线性 LangGraph 负责单轮编排，`messages` 是历史、实际 reasoning 和可观测追踪的唯一业务真相。

**Tech Stack:** Python 3.10+、FastAPI 0.141.1、Uvicorn、LangGraph 1.2.11、langgraph-checkpoint-sqlite 3.1.1、langchain-openai 1.6.0、aiosqlite、原生 HTML/CSS/JavaScript、pytest。

**Spec:** `docs/superpowers/specs/2026-09-03-langgraph-multi-user-chatbot-design.md`

## Global Constraints

- 所有实现必须位于任务专用 Git 工作树，不得覆盖原始 `main` 工作区的未提交内容。
- 旧项目基线固定为 `86220ac52df1641fee051bae8e4e04fd09ff6e40`，其受跟踪内容按原相对路径归档到 `archive/emotion-aware-chatbot-v1/`。
- `.idea/` 留在原工作区；`.pytest_cache/`、`__pycache__/`、`.pyc`、SQLite 运行文件和 `.env` 不进入归档或 Git。
- 核心版本固定为 `fastapi==0.141.1`、`langgraph==1.2.11`、`langgraph-checkpoint-sqlite==3.1.1`、`langchain-openai==1.6.0`。
- 只支持 Python 3.10+、单应用进程和一个 Uvicorn worker。
- 所有业务数据与 Checkpoint 使用同一个 SQLite 文件，并启用 WAL、外键、`busy_timeout=5000` 和 `synchronous=NORMAL`。
- 业务表只能是 `users`、`conversations`、`messages`；业务 Schema 版本使用 `PRAGMA user_version`。
- `messages` 是聊天记录的唯一业务事实来源；Graph State 不保存完整历史、完整 Prompt、reasoning、LLM、连接或密钥。
- 用户标识执行 `strip()` 后长度为 1 到 128，保留大小写并精确匹配；解析后所有接口和隔离使用内部整数 `user_id`。
- 首期每个用户只有一个默认对话，但 `users 1:N conversations` 数据模型不得被写死为一对一。
- 所有用户与助手消息永久保存；模型只加载最近 40 条已完成消息，数量可配置。
- `reasoning_content` 只保存模型实际返回的 reasoning；供应商不返回时必须为 `NULL`，不得生成或推断隐藏思维链。
- Prompt、参数和 trace 必须剔除 API Key、Authorization、Cookie 等密钥；普通 API 和前端不得返回 reasoning、Prompt、参数、trace 或 `thread_id`。
- 客户端断线不能取消后台生成；SSE `done` 只能在助手消息最终提交成功后发送。
- 默认测试使用确定性假模型，不访问真实 LLM；真实模型冒烟测试必须显式启用并单独报告。
- 不实现认证、多对话 UI、情绪识别、长期记忆、向量检索、工具调用、多模型路由或生产级多进程部署。

---

## File Map

### 归档与项目文件

- `archive/emotion-aware-chatbot-v1/**`：旧基线提交的完整受跟踪内容。
- `archive/emotion-aware-chatbot-v1/ARCHIVE.md`：归档提交、范围、参考运行边界。
- `.gitignore`：忽略密钥、环境、缓存、SQLite、任务工作树和视觉草图。
- `.env.example`：新版非密钥配置模板。
- `requirements.txt`：新版直接依赖与核心精确版本。
- `pytest.ini`：测试路径与 asyncio 配置。
- `README.md`：新版安装、运行、数据与安全边界。

### 应用代码

- `chatbot/core/config.py`：环境变量解析和不可变配置。
- `chatbot/core/errors.py`：稳定领域错误码。
- `chatbot/core/time.py`：UTC ISO-8601 时间生成。
- `chatbot/db/connection.py`：aiosqlite 连接、事务和 PRAGMA。
- `chatbot/db/schema.py`：三个业务表和 `user_version` 初始化。
- `chatbot/db/models.py`：User、Conversation、Message、ReservedTurn 数据模型。
- `chatbot/db/users.py`：用户查询与并发安全创建。
- `chatbot/db/conversations.py`：默认对话查询与创建。
- `chatbot/db/messages.py`：消息预留、历史、上下文、流式更新和失败恢复。
- `chatbot/services/identity.py`：同一事务中的用户及默认对话解析。
- `chatbot/llm/types.py`：流式增量、Token 和响应元数据类型。
- `chatbot/llm/redaction.py`：Prompt、参数和 trace 密钥脱敏。
- `chatbot/llm/prompt.py`：从业务消息构造模型 Prompt。
- `chatbot/llm/openai_compatible.py`：ChatOpenAI 创建与流式块归一化。
- `chatbot/graph/state.py`：轻量 `TurnState` 和请求 `TurnContext`。
- `chatbot/graph/dependencies.py`：节点使用的应用生命周期依赖。
- `chatbot/graph/nodes.py`：prepare、generate、finalize 节点。
- `chatbot/graph/builder.py`：线性 StateGraph 构建与编译。
- `chatbot/services/events.py`：内部事件、SSE 序列化和可分离订阅者。
- `chatbot/services/turns.py`：用户锁、幂等、后台任务和图调用协调。
- `chatbot/services/recovery.py`：启动时修复中断消息和相关 Checkpoint。
- `chatbot/api/schemas.py`：请求与公开响应模型。
- `chatbot/api/users.py`：用户解析接口。
- `chatbot/api/messages.py`：历史和流式接口。
- `chatbot/web.py`：FastAPI lifespan、依赖装配、路由和静态资源。
- `chatbot/static/index.html`：用户进入与聊天页面结构。
- `chatbot/static/style.css`：页面样式和响应式布局。
- `chatbot/static/app.js`：用户切换、历史加载、POST SSE 消费和断线校准。
- `chatbot/main.py`：开发启动入口和单 worker 提示。

### 测试代码

- `tests/project/test_archive.py`：归档路径集合；最终验收时增加新版隔离检查。
- `tests/conftest.py`：临时数据库、确定性假模型和应用工厂共享夹具。
- `tests/core/test_config.py`：配置默认值、校验与密钥表示。
- `tests/db/test_schema.py`：DDL、PRAGMA、约束和重启。
- `tests/db/test_identity.py`：并发用户解析和默认对话。
- `tests/db/test_messages.py`：消息生命周期、排序、上下文和幂等。
- `tests/llm/test_redaction.py`：递归密钥脱敏。
- `tests/llm/test_openai_compatible.py`：正文、reasoning、usage 和结束原因解析。
- `tests/graph/test_turn_graph.py`：节点顺序、状态边界和 Checkpoint 隔离。
- `tests/services/test_turns.py`：锁、重放、断线和最终提交顺序。
- `tests/services/test_recovery.py`：进程遗留消息与 Checkpoint 重置。
- `tests/api/test_users.py`：解析接口。
- `tests/api/test_messages.py`：历史字段过滤和 HTTP 错误。
- `tests/api/test_streaming.py`：SSE 顺序、流后错误和断线。
- `tests/app/test_web.py`：生命周期、内部表和静态页面。
- `tests/app/test_acceptance.py`：多用户、重启与端到端假模型验收。
- `tests/app/test_live_llm.py`：显式启用的真实模型冒烟测试。

---

### Task 1: 归档旧项目并建立新版测试骨架

**Files:**
- Move: 基线提交 `86220ac52df1641fee051bae8e4e04fd09ff6e40` 中的所有受跟踪路径 → `archive/emotion-aware-chatbot-v1/<原路径>`
- Create: `archive/emotion-aware-chatbot-v1/ARCHIVE.md`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `tests/project/test_archive.py`

**Interfaces:**
- Consumes: Git 基线提交 `86220ac52df1641fee051bae8e4e04fd09ff6e40`。
- Produces: 冻结归档、可安装的新版 Python 测试环境、干净的根目录和归档完整性测试。

- [ ] **Step 1: 使用旧基线依赖建立测试运行器**

Run: `/opt/homebrew/bin/python3.12 -m venv .venv`

Run: `.venv/bin/python -m pip install -r requirements.txt`

Run: `.venv/bin/python -m pytest --version`

Expected: 输出 pytest 版本并以状态 0 结束。此时安装使用的仍是尚未归档的旧 `requirements.txt`，只用于保证 RED 测试能够实际运行。

- [ ] **Step 2: 写归档完整性失败测试**

```python
# tests/project/test_archive.py
from pathlib import Path
import subprocess

BASELINE = "86220ac52df1641fee051bae8e4e04fd09ff6e40"
ARCHIVE = Path("archive/emotion-aware-chatbot-v1")


def test_every_baseline_file_exists_under_archive():
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE], text=True
    )
    missing = [path for path in output.splitlines() if not (ARCHIVE / path).is_file()]
    assert missing == []
```

- [ ] **Step 3: 运行测试并确认归档尚未完成**

Run: `.venv/bin/python -m pytest tests/project/test_archive.py -v`

Expected: FAIL，至少报告 `archive/emotion-aware-chatbot-v1/<旧路径>` 不存在。

- [ ] **Step 4: 按基线文件清单逐文件移动旧内容**

从工作树根目录执行以下逻辑；必须逐文件 `git mv`，不能整体移动目录，以免把被忽略的缓存带入归档：

```bash
baseline=86220ac52df1641fee051bae8e4e04fd09ff6e40
archive_root=archive/emotion-aware-chatbot-v1
while IFS= read -r path; do
  mkdir -p "$archive_root/$(dirname "$path")"
  git mv "$path" "$archive_root/$path"
done < <(git ls-tree -r --name-only "$baseline")
```

设计文档和本实施计划不属于基线提交，因此仍保留在根目录 `docs/superpowers/`。

- [ ] **Step 5: 清理未归档的生成缓存并创建归档说明**

使用可恢复方式移除旧路径中遗留的缓存；不得移动或删除 `.idea/`：

```bash
find chatbot scripts tests -type d -name __pycache__ -prune -exec /usr/bin/trash {} +
test ! -d .pytest_cache || /usr/bin/trash .pytest_cache
```

`ARCHIVE.md` 必须写明原始提交、归档日期、包含范围、从归档目录运行旧项目的命令，以及“仅供参考、不与新版共享运行时代码”。

- [ ] **Step 6: 创建新版根配置**

`requirements.txt` 使用：

```text
fastapi==0.141.1
uvicorn>=0.30,<1
langgraph==1.2.11
langgraph-checkpoint-sqlite==3.1.1
langchain-openai==1.6.0
aiosqlite>=0.20,<1
python-dotenv>=1.0,<2
httpx>=0.27,<1
pytest>=8,<10
pytest-asyncio>=0.25,<2
```

`pytest.ini` 使用 `pythonpath = .`、`testpaths = tests`、`asyncio_mode = auto`。根 `.gitignore` 至少忽略 `.env`、`.venv/`、`.idea/`、`.pytest_cache/`、`__pycache__/`、`*.pyc`、`.worktrees/`、`.superpowers/`、`data/*.sqlite3*`。

- [ ] **Step 7: 安装新版依赖并运行归档测试**

Run: `.venv/bin/python -m pip install -r requirements.txt`

Run: `.venv/bin/python -m pytest tests/project/test_archive.py -v`

Expected: PASS；归档缺失列表为空。

- [ ] **Step 8: 核对移动范围并提交**

Run: `git status --short`

Run: `git diff --summary`

Expected: 旧基线路径全部显示为移动到统一归档前缀；设计与计划仍在根 `docs/superpowers/`。

```bash
git add -A
git commit -m "chore: archive legacy emotion chatbot"
```

---

### Task 2: 实现配置、SQLite 连接与业务 Schema

**Files:**
- Create: `chatbot/__init__.py`
- Create: `chatbot/core/__init__.py`
- Create: `chatbot/core/config.py`
- Create: `chatbot/core/errors.py`
- Create: `chatbot/core/time.py`
- Create: `chatbot/db/__init__.py`
- Create: `chatbot/db/connection.py`
- Create: `chatbot/db/schema.py`
- Create: `chatbot/db/models.py`
- Create: `.env.example`
- Create: `tests/conftest.py`
- Test: `tests/core/test_config.py`
- Test: `tests/db/test_schema.py`

**Interfaces:**
- Consumes: Task 1 的依赖和根配置。
- Produces: `AppConfig.from_env() -> AppConfig`、`Database.connect()`、`Database.transaction(immediate: bool)`、`initialize_schema(database)`、User/Conversation/Message/InterruptedTurn 数据模型。

- [ ] **Step 1: 写配置和 Schema 失败测试**

```python
def test_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        AppConfig.from_env()


@pytest.mark.asyncio
async def test_schema_creates_three_business_tables(tmp_path):
    database = Database(tmp_path / "chatbot.sqlite3")
    await initialize_schema(database)
    async with database.connect() as conn:
        names = {
            row[0]
            for row in await (await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )).fetchall()
        }
        version = (await (await conn.execute("PRAGMA user_version")).fetchone())[0]
    assert {"users", "conversations", "messages"} <= names
    assert version == 1
```

- [ ] **Step 2: 运行测试并确认模块不存在**

Run: `.venv/bin/python -m pytest tests/core/test_config.py tests/db/test_schema.py -v`

Expected: FAIL with `ModuleNotFoundError` for new modules.

- [ ] **Step 3: 实现配置与稳定错误类型**

实现不可变 `AppConfig`，字段固定为：

```python
@dataclass(frozen=True)
class AppConfig:
    llm_api_key: SecretStr
    llm_model: str
    llm_base_url: str | None
    llm_temperature: float
    llm_timeout_seconds: float
    context_message_limit: int
    sqlite_db_path: Path
```

类方法签名固定为 `AppConfig.from_env() -> AppConfig`，逐项读取上述七个字段并执行下一段的范围校验。

校验温度 `0..2`、超时为正、上下文条数 `1..200`。`repr(config)` 不得包含 API Key。`DomainError` 包含 `code`、`message`、`http_status`，首批错误码为 `invalid_identifier`、`invalid_message`、`user_not_found`、`turn_in_progress`、`model_error`、`database_error`、`process_interrupted`。

- [ ] **Step 4: 实现连接、事务和数据模型**

`Database` 每次连接执行四个 PRAGMA；`transaction(immediate=True)` 执行 `BEGIN IMMEDIATE`，异常回滚，成功提交。定义 `User`、`Conversation`、`Message`、`ReservedTurn`、`InterruptedTurn(conversation_id, thread_id, assistant_message_id)` 冻结 dataclass，并使用 UTC ISO-8601 文本时间。

- [ ] **Step 5: 实现设计文档中的精确 DDL**

`initialize_schema(database)` 必须：

1. 读取 `PRAGMA user_version`。
2. 版本 `0` 时在一个立即事务中创建 `users`、`conversations`、`messages` 及设计中的索引。
3. 设置 `PRAGMA user_version = 1`。
4. 版本 `1` 时幂等返回。
5. 版本大于 `1` 时抛出 `ConfigError("unsupported database schema version")`。

- [ ] **Step 6: 运行测试并检查约束**

Run: `.venv/bin/python -m pytest tests/core/test_config.py tests/db/test_schema.py -v`

Expected: PASS，包括 WAL、外键、唯一约束、CHECK 约束、重启幂等和不支持版本测试。

- [ ] **Step 7: 提交基础设施**

```bash
git add .env.example chatbot/core chatbot/db tests/core tests/db/test_schema.py
git commit -m "feat: add chatbot configuration and sqlite schema"
```

---

### Task 3: 实现用户解析与可扩展默认对话

**Files:**
- Create: `chatbot/db/users.py`
- Create: `chatbot/db/conversations.py`
- Create: `chatbot/services/__init__.py`
- Create: `chatbot/services/identity.py`
- Test: `tests/db/test_identity.py`

**Interfaces:**
- Consumes: `Database.transaction()`、`User`、`Conversation`、UTC clock。
- Produces: `UserRepository.get_or_create(conn, identifier) -> User`、`ConversationRepository.get_or_create_default(conn, user_id) -> Conversation`、`ConversationRepository.get_default_by_user(user_id) -> Conversation | None`、`IdentityService.resolve(identifier) -> ResolvedIdentity`。

- [ ] **Step 1: 写精确匹配、自动创建和并发失败测试**

```python
@pytest.mark.asyncio
async def test_resolve_reuses_trimmed_identifier(identity_service):
    first = await identity_service.resolve("  Alice  ")
    second = await identity_service.resolve("Alice")
    assert first.user.id == second.user.id
    assert first.conversation.thread_id == second.conversation.thread_id


@pytest.mark.asyncio
async def test_identifier_is_case_sensitive(identity_service):
    upper, lower = await asyncio.gather(
        identity_service.resolve("Alice"),
        identity_service.resolve("alice"),
    )
    assert upper.user.id != lower.user.id


@pytest.mark.asyncio
async def test_concurrent_resolve_creates_one_default(identity_service, database):
    results = await asyncio.gather(
        *(identity_service.resolve("shared") for _ in range(8))
    )
    assert len({item.user.id for item in results}) == 1
    assert len({item.conversation.id for item in results}) == 1
```

- [ ] **Step 2: 运行测试并确认仓储缺失**

Run: `.venv/bin/python -m pytest tests/db/test_identity.py -v`

Expected: FAIL with missing repository/service imports.

- [ ] **Step 3: 实现用户和默认对话仓储**

`UserRepository` 在调用者事务内执行以下语句，避免无保护的先查后插：

```sql
INSERT INTO users(identifier, created_at, updated_at)
VALUES (?, ?, ?)
ON CONFLICT(identifier) DO UPDATE SET updated_at = users.updated_at
RETURNING id, identifier, created_at, updated_at;
```

`ConversationRepository` 先查询 `is_default=1`；不存在时插入 UUID4 `id` 和独立 UUID4 `thread_id`，遇到部分唯一索引冲突后重查。`get_default_by_user()` 只按内部 `user_id` 返回默认对话，不接受外部 identifier。

- [ ] **Step 4: 实现身份服务事务边界**

```python
@dataclass(frozen=True)
class ResolvedIdentity:
    user: User
    conversation: Conversation


class IdentityService:
    async def resolve(self, identifier: str) -> ResolvedIdentity:
        normalized = normalize_identifier(identifier)
        async with self.database.transaction(immediate=True) as conn:
            user = await self.users.get_or_create(conn, normalized)
            conversation = await self.conversations.get_or_create_default(conn, user.id)
        return ResolvedIdentity(user=user, conversation=conversation)
```

- [ ] **Step 5: 运行身份测试**

Run: `.venv/bin/python -m pytest tests/db/test_identity.py tests/db/test_schema.py -v`

Expected: PASS；8 个并发解析只产生一名用户和一个默认对话，未来仍可插入 `is_default=0` 的额外对话。

- [ ] **Step 6: 提交身份能力**

```bash
git add chatbot/db/users.py chatbot/db/conversations.py chatbot/services tests/db/test_identity.py
git commit -m "feat: resolve users and default conversations"
```

---

### Task 4: 实现消息事实仓储与完整生命周期

**Files:**
- Create: `chatbot/db/messages.py`
- Test: `tests/db/test_messages.py`

**Interfaces:**
- Consumes: `Database`、`Message`、`ReservedTurn`、UTC clock。
- Produces: `reserve_turn()`、`find_turn()`、`find_assistant()`、`list_visible()`、`list_context()`、`mark_streaming()`、`flush_partial()`、`complete_assistant()`、`fail_assistant()`、`fail_interrupted()`。

- [ ] **Step 1: 写消息预留与幂等失败测试**

```python
@pytest.mark.asyncio
async def test_reserve_turn_writes_pair_with_stable_sequence(messages, conversation):
    turn = await messages.reserve_turn(conversation.id, REQUEST_ID, "你好")
    assert turn.user.sequence_no == 1
    assert turn.user.status == "completed"
    assert turn.assistant.sequence_no == 2
    assert turn.assistant.status == "pending"
    assert turn.user.request_id == turn.assistant.request_id == REQUEST_ID


@pytest.mark.asyncio
async def test_reserve_same_request_returns_existing_pair(messages, conversation):
    first = await messages.reserve_turn(conversation.id, REQUEST_ID, "你好")
    second = await messages.reserve_turn(conversation.id, REQUEST_ID, "不同内容")
    assert second == first
```

- [ ] **Step 2: 写历史、上下文和失败状态测试**

验证：

- 可见历史按 `sequence_no` 正序分页。
- 上下文只返回 `completed` 用户消息和 `completed` 助手消息。
- `failed` 助手消息保留部分正文和 reasoning，但不进入 Prompt。
- `flush_partial()` 同时更新正文、reasoning、trace 和 `updated_at`。
- `complete_assistant()` 原子保存最终正文、reasoning、Prompt、参数、Token、耗时和结束原因。
- `fail_interrupted()` 返回受影响的 `conversation_id` 与 `thread_id` 集合。

- [ ] **Step 3: 运行测试并确认实现缺失**

Run: `.venv/bin/python -m pytest tests/db/test_messages.py -v`

Expected: FAIL with missing `MessageRepository`.

- [ ] **Step 4: 实现预留和查询**

`reserve_turn(conversation_id, request_id, content)` 在 `BEGIN IMMEDIATE` 内先按请求查询既有 pair；不存在时以 `MAX(sequence_no)+1` 分配两个连续序号并同时插入。`list_visible()` 不选择 reasoning、Prompt、参数或 trace；`list_context()` 接受明确的 `limit` 并过滤状态。

- [ ] **Step 5: 实现状态迁移保护**

所有更新 SQL 必须包含当前状态条件，例如：

```sql
UPDATE messages
SET status = 'streaming', updated_at = ?
WHERE id = ? AND role = 'assistant' AND status = 'pending';
```

更新影响行数不是 1 时抛出 `InvalidMessageState`。`completed` 和 `failed` 为终态，不能被普通流式更新覆盖。

- [ ] **Step 6: 运行消息测试**

Run: `.venv/bin/python -m pytest tests/db/test_messages.py tests/db/test_identity.py -v`

Expected: PASS，包括并发序号、幂等 pair、状态迁移、字段完整性和分页。

- [ ] **Step 7: 提交消息仓储**

```bash
git add chatbot/db/messages.py tests/db/test_messages.py
git commit -m "feat: persist complete chat message lifecycle"
```

---

### Task 5: 实现 Prompt、密钥脱敏与 OpenAI-compatible 流

**Files:**
- Create: `chatbot/llm/__init__.py`
- Create: `chatbot/llm/types.py`
- Create: `chatbot/llm/redaction.py`
- Create: `chatbot/llm/prompt.py`
- Create: `chatbot/llm/openai_compatible.py`
- Test: `tests/llm/test_redaction.py`
- Test: `tests/llm/test_openai_compatible.py`

**Interfaces:**
- Consumes: `AppConfig`、上下文 `Message` 列表。
- Produces: `build_prompt(messages) -> list[BaseMessage]`、`redact_secrets(value) -> JSONValue`、`ChatModelAdapter.stream(prompt) -> AsyncIterator[ModelDelta]`。

- [ ] **Step 1: 写递归脱敏和流式块失败测试**

```python
def test_redaction_removes_sensitive_keys_recursively():
    value = {
        "Authorization": "Bearer secret",
        "nested": {"api_key": "secret", "temperature": 0.7},
        "cookie": "session=secret",
    }
    assert redact_secrets(value) == {
        "Authorization": "[REDACTED]",
        "nested": {"api_key": "[REDACTED]", "temperature": 0.7},
        "cookie": "[REDACTED]",
    }


def test_parse_chunk_keeps_actual_reasoning_separate():
    chunk = AIMessageChunk(
        content="答案",
        additional_kwargs={"reasoning_content": "依据"},
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    )
    delta = parse_ai_message_chunk(chunk)
    assert delta.content == "答案"
    assert delta.reasoning == "依据"
    assert delta.usage.total_tokens == 5
```

- [ ] **Step 2: 运行测试并确认模块缺失**

Run: `.venv/bin/python -m pytest tests/llm -v`

Expected: FAIL with missing new LLM modules.

- [ ] **Step 3: 定义稳定流式类型**

```python
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelDelta:
    content: str = ""
    reasoning: str = ""
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    response_metadata: dict[str, object] = field(default_factory=dict)
```

- [ ] **Step 4: 实现 Prompt 与脱敏**

Prompt 固定包含一个版本化系统提示和数据库返回的上下文消息。序列化存储前对字典和列表递归脱敏；敏感键比较忽略大小写及 `-`/`_` 差异，覆盖 `api_key`、`authorization`、`cookie`、`set_cookie`、`access_token`、`secret`。

- [ ] **Step 5: 实现 ChatOpenAI 适配**

`OpenAICompatibleChatModel` 使用以下参数创建底层客户端：

```python
ChatOpenAI(
    api_key=config.llm_api_key,
    model=config.llm_model,
    base_url=config.llm_base_url,
    temperature=config.llm_temperature,
    timeout=config.llm_timeout_seconds,
    streaming=True,
)
```

`stream()` 调用 `astream()`，正文从标准 content 或文本 content block 提取；reasoning 只从供应商真实字段提取，缺失时返回空增量，最终数据库值保持 `NULL`。

- [ ] **Step 6: 运行 LLM 单元测试**

Run: `.venv/bin/python -m pytest tests/llm -v`

Expected: PASS；测试不访问网络，密钥不会出现在 `repr`、Prompt JSON、参数 JSON 或 trace。

- [ ] **Step 7: 提交模型边界**

```bash
git add chatbot/llm tests/llm
git commit -m "feat: add redacted openai-compatible streaming adapter"
```

---

### Task 6: 实现轻量持久化 LangGraph

**Files:**
- Create: `chatbot/graph/__init__.py`
- Create: `chatbot/graph/state.py`
- Create: `chatbot/graph/dependencies.py`
- Create: `chatbot/graph/nodes.py`
- Create: `chatbot/graph/builder.py`
- Test: `tests/graph/test_turn_graph.py`

**Interfaces:**
- Consumes: `MessageRepository`、`ChatModelAdapter`、事件发布协议、`AsyncSqliteSaver`。
- Produces: `EventPublisher`、`TurnContext`、`build_turn_graph(deps) -> StateGraph`、`compile_turn_graph(deps, checkpointer) -> CompiledStateGraph`。

- [ ] **Step 1: 写图结构和状态边界失败测试**

```python
def test_turn_state_contains_only_control_fields():
    assert set(TurnState.__annotations__) == {
        "user_id",
        "conversation_id",
        "request_id",
        "user_message_id",
        "assistant_message_id",
        "phase",
        "error_code",
    }


@pytest.mark.asyncio
async def test_graph_runs_nodes_in_order(compiled_graph, turn_context, messages):
    state = {
        "user_id": turn_context.user_id,
        "conversation_id": turn_context.conversation_id,
        "request_id": turn_context.request_id,
        "user_message_id": turn_context.user_message_id,
        "assistant_message_id": turn_context.assistant_message_id,
        "phase": "reserved",
        "error_code": None,
    }
    result = await compiled_graph.ainvoke(
        state,
        {"configurable": {"thread_id": turn_context.thread_id}},
        context=turn_context,
    )
    assert result["phase"] == "completed"
    assistant = await messages.find_assistant(
        turn_context.conversation_id, turn_context.request_id
    )
    trace = json.loads(assistant.trace_json)
    assert [node["name"] for node in trace["nodes"]] == [
        "prepare_turn", "generate_response", "finalize_turn"
    ]
```

- [ ] **Step 2: 运行测试并确认图模块缺失**

Run: `.venv/bin/python -m pytest tests/graph/test_turn_graph.py -v`

Expected: FAIL with missing `TurnState` and graph builder.

- [ ] **Step 3: 定义状态、上下文和依赖**

`TurnState` 字段必须与测试完全一致。请求上下文和发布协议固定为：

```python
class EventPublisher(Protocol):
    async def publish(self, name: str, data: dict[str, object]) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class TurnContext:
    thread_id: str
    user_id: int
    conversation_id: str
    request_id: str
    user_message_id: str
    assistant_message_id: str
    publisher: EventPublisher
```

`NodeDependencies` 包含消息仓储、模型适配、上下文上限、时钟与节流配置。连接、LLM 和发布器均不得出现在 `TurnState`。

- [ ] **Step 4: 实现三个幂等节点**

- `prepare_turn` 仅在助手为 `pending` 时改为 `streaming`；已经 `completed` 时安全返回完成状态。
- `generate_response` 查询当前对话最近已完成消息，构造并保存 Prompt，流式聚合正文/reasoning/usage，每 250ms 或新增 256 字符执行一次 `flush_partial()`。
- `finalize_turn` 成功时调用 `complete_assistant()`；模型错误时调用 `fail_assistant()`，并把稳定错误码写入状态。

节点恢复执行时先读取消息终态，避免 Checkpoint 重放导致第二次模型调用或覆盖已完成消息。

- [ ] **Step 5: 构建并编译线性图**

```python
builder = StateGraph(TurnState, context_schema=TurnContext)
builder.add_node("prepare_turn", prepare_turn)
builder.add_node("generate_response", generate_response)
builder.add_node("finalize_turn", finalize_turn)
builder.add_edge(START, "prepare_turn")
builder.add_edge("prepare_turn", "generate_response")
builder.add_edge("generate_response", "finalize_turn")
builder.add_edge("finalize_turn", END)
```

编译时必须传入官方 `AsyncSqliteSaver`。

- [ ] **Step 6: 验证 SQLite Checkpoint 隔离**

增加集成测试：两个 UUID `thread_id` 分别运行后，`checkpoints` 中均有记录；`graph.aget_state()` 返回各自状态；两个 snapshot 的 `values` 键集合都严格等于 `TurnState` 字段集合，且 `repr(snapshot.values)` 不包含 API Key 或完整历史消息数组。

Run: `.venv/bin/python -m pytest tests/graph/test_turn_graph.py -v`

Expected: PASS，节点顺序、异常收尾、恢复幂等和线程隔离全部通过。

- [ ] **Step 7: 提交 LangGraph**

```bash
git add chatbot/graph tests/graph
git commit -m "feat: add persistent lightweight turn graph"
```

---

### Task 7: 实现后台生成、用户锁、幂等重放与 SSE 事件

**Files:**
- Create: `chatbot/services/events.py`
- Create: `chatbot/services/turns.py`
- Test: `tests/services/test_turns.py`

**Interfaces:**
- Consumes: `ConversationRepository.get_default_by_user()`、`MessageRepository`、编译后的图、`TurnContext`、`EventPublisher`。
- Produces: `TurnCoordinator.open_stream(user_id, request_id, content) -> TurnSubscription`、`TurnSubscription.events() -> AsyncIterator[SseEvent]`、`TurnCoordinator.shutdown(timeout_seconds)`。

- [ ] **Step 1: 写事件顺序和提交门槛失败测试**

```python
@pytest.mark.asyncio
async def test_done_is_emitted_after_database_completion(coordinator, messages):
    subscription = await coordinator.open_stream(1, REQUEST_ID, "你好")
    events = [event async for event in subscription.events()]
    assistant = await messages.find_assistant(CONVERSATION_ID, REQUEST_ID)
    assert [event.name for event in events] == [
        "run_started", "user_message", "token", "done"
    ]
    assert assistant.status == "completed"
    assert events[-1].data["message"]["id"] == assistant.id
```

- [ ] **Step 2: 写锁、重放和断线失败测试**

验证：

- 同一用户已有运行任务时第二个不同请求抛出 `turn_in_progress`。
- 不同用户的假模型屏障证明两个生成同时进入。
- 已完成 `request_id` 直接发出重放事件，模型调用次数不增加。
- 订阅者调用 `detach()` 后，后台任务继续完成消息，发布器不再累计事件。
- `shutdown()` 等待任务；超时任务被取消并标记失败。

- [ ] **Step 3: 运行测试并确认协调器缺失**

Run: `.venv/bin/python -m pytest tests/services/test_turns.py -v`

Expected: FAIL with missing coordinator/event modules.

- [ ] **Step 4: 实现事件与可分离订阅**

```python
@dataclass(frozen=True)
class SseEvent:
    name: Literal["run_started", "user_message", "token", "done", "error"]
    data: dict[str, object]
```

`TurnSubscription.events() -> AsyncIterator[SseEvent]` 从内部队列读取直到终止哨兵；`TurnSubscription.detach() -> None` 原子地把订阅标记为已分离并清空未消费事件。

内部队列设置容量上限；`detach()` 后发布器丢弃仅用于浏览器的事件，但后台消息持久化不受影响。

- [ ] **Step 5: 实现协调器**

执行顺序固定为：获取用户锁 → 检查 `request_id` → 预留消息 pair → 建立订阅 → 创建后台任务 → `graph.ainvoke()`。模型节点通过 `EventPublisher` 发布正文 Token，因此协调器不再从第二套 LangGraph 事件流重复提取 Token。协调器维护 `dict[int, asyncio.Lock]` 和后台任务集合；任务完成回调移除集合并释放锁。

- [ ] **Step 6: 实现幂等重放**

已完成请求发送 `run_started` 和 `done(replayed=true)`，不重复插入消息或调用模型。已失败请求发送保存的 `error`；调用方若需重新生成必须创建新 UUID。

- [ ] **Step 7: 运行服务测试**

Run: `.venv/bin/python -m pytest tests/services/test_turns.py tests/graph/test_turn_graph.py -v`

Expected: PASS；测试显式断言 `done` 发生在数据库终态之后。

- [ ] **Step 8: 提交协调与 SSE 领域事件**

```bash
git add chatbot/services/events.py chatbot/services/turns.py tests/services
git commit -m "feat: coordinate resilient streaming turns"
```

---

### Task 8: 实现 FastAPI、生命周期、恢复与公开 API

**Files:**
- Create: `chatbot/api/__init__.py`
- Create: `chatbot/api/schemas.py`
- Create: `chatbot/api/users.py`
- Create: `chatbot/api/messages.py`
- Create: `chatbot/services/recovery.py`
- Create: `chatbot/web.py`
- Create: `chatbot/main.py`
- Test: `tests/api/test_users.py`
- Test: `tests/api/test_messages.py`
- Test: `tests/api/test_streaming.py`
- Test: `tests/services/test_recovery.py`
- Test: `tests/app/test_web.py`

**Interfaces:**
- Consumes: Tasks 2-7 的配置、数据库、身份、消息、图和协调器。
- Produces: `create_app(config: AppConfig | None = None, model: ChatModelAdapter | None = None) -> FastAPI` 与三个设计 API。

- [ ] **Step 1: 写用户解析 API 失败测试**

```python
def test_resolve_creates_and_reuses_user(client):
    first = client.post("/api/users/resolve", json={"identifier": " Alice "})
    second = client.post("/api/users/resolve", json={"identifier": "Alice"})
    assert first.status_code == 200
    assert second.json()["user"]["id"] == first.json()["user"]["id"]
    assert "thread_id" not in first.text
```

- [ ] **Step 2: 写历史字段过滤与 SSE 失败测试**

```python
def test_history_hides_internal_fields(client, seeded_user):
    response = client.get(f"/api/users/{seeded_user.id}/messages")
    assert response.status_code == 200
    serialized = response.text
    for forbidden in ("reasoning_content", "prompt_json", "trace_json", "thread_id"):
        assert forbidden not in serialized
```

SSE 测试必须解析事件帧并断言 `run_started → user_message → token* → done|error`；头部发送后的异常只使用 `error` 事件，不再尝试修改 HTTP 状态。

- [ ] **Step 3: 运行 API 测试并确认应用缺失**

Run: `.venv/bin/python -m pytest tests/api tests/services/test_recovery.py tests/app/test_web.py -v`

Expected: FAIL with missing FastAPI app and routes.

- [ ] **Step 4: 实现 Pydantic Schema 与用户路由**

请求模型：

```python
class ResolveUserRequest(BaseModel):
    identifier: str


class SendMessageRequest(BaseModel):
    request_id: UUID
    content: str
```

公开消息响应只包含设计允许字段。领域错误通过统一 handler 转为 `{ "error": {"code": "stable_code", "message": "safe message"} }`。

- [ ] **Step 5: 实现历史与 POST SSE 路由**

历史参数校验 `limit=1..200`，默认 100；`before_sequence` 必须为正整数。流式接口使用 Fetch 可消费的 POST `StreamingResponse(media_type="text/event-stream")`，并设置 `Cache-Control: no-cache` 和 `X-Accel-Buffering: no`。

- [ ] **Step 6: 实现启动恢复**

```python
async def recover_interrupted_turns(
    messages: MessageRepository,
    checkpointer: AsyncSqliteSaver,
) -> RecoveryReport:
    interrupted = await messages.fail_interrupted(error_code="process_interrupted")
    thread_ids = sorted({item.thread_id for item in interrupted})
    for thread_id in thread_ids:
        await checkpointer.adelete_thread(thread_id)
    return RecoveryReport(
        failed_messages=len(interrupted),
        reset_threads=len(thread_ids),
    )
```

恢复只重置存在遗留 `pending/streaming` 消息的线程；正常线程 Checkpoint 保留。

- [ ] **Step 7: 实现 FastAPI lifespan**

启动顺序：加载配置 → 初始化业务 Schema → 打开独立 aiosqlite Checkpointer 连接并统一 PRAGMA → `await checkpointer.setup()` → 验证内部表包含 Saver 需要的 `checkpoints`、`writes` → 运行恢复 → 编译图 → 创建协调器。关闭顺序：停止新请求 → `coordinator.shutdown()` → 关闭 Checkpointer 和业务资源。

- [ ] **Step 8: 运行 API、恢复与内部表测试**

Run: `.venv/bin/python -m pytest tests/api tests/services/test_recovery.py tests/app/test_web.py -v`

Expected: PASS；SQLite 中存在三个业务表和 `checkpoints`、`writes`，重启后正常历史可读取，遗留运行状态被标记失败。

- [ ] **Step 9: 提交 Web API**

```bash
git add chatbot/api chatbot/services/recovery.py chatbot/web.py chatbot/main.py tests/api tests/services/test_recovery.py tests/app/test_web.py
git commit -m "feat: expose persistent multi-user chat api"
```

---

### Task 9: 实现原生单页多用户聊天界面

**Files:**
- Create: `chatbot/static/index.html`
- Create: `chatbot/static/style.css`
- Create: `chatbot/static/app.js`
- Modify: `chatbot/web.py`
- Test: `tests/app/test_web.py`

**Interfaces:**
- Consumes: `/api/users/resolve`、`/api/users/{user_id}/messages`、`/api/users/{user_id}/messages:stream`。
- Produces: 用户标识进入、切换用户、历史加载、POST SSE 流式展示与断线校准页面。

- [ ] **Step 1: 写静态页面契约失败测试**

```python
def test_index_contains_required_controls(client):
    html = client.get("/").text
    for element_id in (
        "identity-form", "identifier-input", "chat-view",
        "switch-user", "message-list", "message-form", "message-input",
    ):
        assert f'id="{element_id}"' in html
```

增加 JavaScript 静态断言：代码使用 `fetch()` POST 流、`ReadableStream` 解帧、`crypto.randomUUID()` request ID，并且不存在 `reasoning_content`、`prompt_json` 或 `trace_json` 展示逻辑。

- [ ] **Step 2: 运行页面测试并确认资源缺失**

Run: `.venv/bin/python -m pytest tests/app/test_web.py -v`

Expected: FAIL，首页或必需控件不存在。

- [ ] **Step 3: 实现语义化 HTML 与响应式样式**

页面只有两个视图：用户标识进入视图和聊天视图。聊天视图包含当前 identifier、切换按钮、消息列表、状态区域、输入框和发送按钮。为 `pending`、`streaming`、`completed`、`failed` 提供可辨识但不泄露内部错误的状态样式。

- [ ] **Step 4: 实现用户解析与历史加载**

提交 identifier 后保存当前页面会话中的 `{user_id, identifier}`，调用历史 API 并按 `sequence_no` 渲染。切换用户清除当前页面状态并返回输入视图；不把该流程描述为登录。

- [ ] **Step 5: 实现 POST SSE 解析**

`app.js` 必须处理跨 chunk 的 SSE 行缓冲，按空行分帧并解析 `event:`/`data:`。`token` 只追加正文；`done` 用服务器最终消息替换临时内容；`error` 显示稳定提示。发送期间禁用输入与按钮。

- [ ] **Step 6: 实现断线数据库校准**

流异常或页面消费提前结束时，在短暂延迟后重新调用历史 API，以数据库状态替换临时消息。前端不得因为断线再次自动提交相同正文。

- [ ] **Step 7: 运行页面与 API 测试**

Run: `.venv/bin/python -m pytest tests/app/test_web.py tests/api -v`

Expected: PASS；首页可加载，静态资源返回正确 MIME，内部字段不出现在 DOM 或脚本渲染路径。

- [ ] **Step 8: 提交前端**

```bash
git add chatbot/static chatbot/web.py tests/app/test_web.py
git commit -m "feat: add multi-user streaming chat interface"
```

---

### Task 10: 完成多用户、重启和真实模型分层验收

**Files:**
- Create: `tests/app/test_acceptance.py`
- Create: `tests/app/test_live_llm.py`
- Create: `README.md`
- Modify: `.env.example`
- Modify: `archive/emotion-aware-chatbot-v1/ARCHIVE.md`
- Modify: `tests/project/test_archive.py`

**Interfaces:**
- Consumes: 完整应用及所有公开接口。
- Produces: 可复制运行说明、确定性验收证据和可选真实模型冒烟入口。

- [ ] **Step 1: 写多用户端到端验收测试**

```python
def test_two_users_keep_history_and_context_isolated(app_factory, fake_model):
    with app_factory() as client:
        alice = client.post("/api/users/resolve", json={"identifier": "alice"}).json()
        bob = client.post("/api/users/resolve", json={"identifier": "bob"}).json()
        stream_message(client, alice["user"]["id"], "Alice secret")
        stream_message(client, bob["user"]["id"], "Bob secret")
        assert "Bob secret" not in fake_model.prompts_for(alice["user"]["id"])
        assert "Alice secret" not in fake_model.prompts_for(bob["user"]["id"])
```

- [ ] **Step 2: 写同一数据库重启验收测试**

第一次应用生命周期创建用户并完成消息；关闭后使用同一 SQLite 路径创建第二个应用，断言用户 ID、默认对话、完整消息、reasoning、trace 和正常 Checkpoint 均存在。另一个用例预置 `streaming` 助手消息，重启后断言其为 `failed/process_interrupted` 且对应线程 Checkpoint 已重置。

在 `tests/project/test_archive.py` 增加非空源文件扫描，确保新版不导入归档：

```python
def test_new_runtime_does_not_import_archive():
    sources = list(Path("chatbot").rglob("*.py"))
    assert sources
    offenders = [path for path in sources if "archive." in path.read_text()]
    assert offenders == []
```

- [ ] **Step 3: 写显式真实模型冒烟测试**

```python
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TEST") != "1",
    reason="set RUN_LIVE_LLM_TEST=1 to call the configured model",
)


def test_live_model_completes_and_persists_trace(live_client):
    user_id = resolve(live_client, "live-smoke")
    events = stream_message(live_client, user_id, "只回复 OK")
    assert events[-1]["event"] == "done"
```

该测试不得断言供应商一定返回 reasoning；只在返回时验证数据库原样保存。

- [ ] **Step 4: 运行完整离线测试**

Run: `.venv/bin/python -m pytest -q`

Expected: 所有离线测试 PASS；真实模型测试显示 SKIPPED，而不是伪装为真实调用成功。

- [ ] **Step 5: 编写 README 与配置模板**

README 必须包含：

- 旧项目归档位置及基线提交。
- Python 3.10+ 创建 `.venv` 和安装命令。
- `.env.example` 字段说明。
- `uvicorn chatbot.web:app --workers 1 --no-access-log` 启动命令。
- 用户标识不是认证、同标识可访问同一数据的警告。
- 单进程 SQLite 边界、数据文件位置和明文敏感聊天数据提示。
- 三个业务表与两个官方 Checkpointer 表的职责。
- reasoning 只保存供应商实际返回值且不在前端展示。
- 离线测试和显式真实模型测试命令。

- [ ] **Step 6: 执行真实模型冒烟测试（仅在凭据可用时）**

Run: `RUN_LIVE_LLM_TEST=1 .venv/bin/python -m pytest tests/app/test_live_llm.py -v`

Expected with valid credentials: PASS，并在 SQLite 中产生完成消息和 trace。若未提供凭据，不执行此命令，验收报告明确写“未进行真实模型调用”。

- [ ] **Step 7: 执行最终仓库验证**

Run: `git status --short`

Run: `git diff --check`

Run: `git ls-files | rg '(^|/)\.env$|(^|/)__pycache__/|(^|/)\.pytest_cache/|\.sqlite3($|[-.])'`

Expected: 工作树只含本任务预期变更；diff 无空白错误；密钥、缓存和 SQLite 运行文件搜索无输出。

Run: `.venv/bin/python -m pytest -q`

Expected: 离线套件全部通过。

- [ ] **Step 8: 提交验收与文档**

```bash
git add README.md .env.example archive/emotion-aware-chatbot-v1/ARCHIVE.md tests/app tests/project/test_archive.py
git commit -m "docs: complete chatbot rebuild verification"
```

---

## Final Review Checklist

- [ ] `git status --short --branch` 只显示干净任务分支。
- [ ] `git log --oneline <target>..HEAD` 中每个任务提交边界清楚。
- [ ] `git diff --check main..HEAD` 通过。
- [ ] 归档完整性测试通过，旧基线受跟踪文件集合无缺失。
- [ ] 三个业务表和官方 `checkpoints`、`writes` 在同一个 SQLite 文件中。
- [ ] 两用户隔离、同用户连续上下文、幂等重放和断线后台完成均有自动化证据。
- [ ] 每条助手回复保存正文、可用 reasoning、Prompt、参数、Token、耗时和 trace。
- [ ] 普通历史 API 和前端不返回 reasoning、Prompt、trace 或 `thread_id`。
- [ ] 默认测试没有发起网络模型调用；真实模型验证状态单独报告。
- [ ] 未实现任何首期范围外功能。
- [ ] 在用户明确批准前不合并、不推送、不删除任务分支或工作树。
