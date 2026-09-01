# HDU-erc LangGraph Conversation Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 HDU-erc 在线聊天从 `ChatService + RunnableWithMessageHistory` 迁移为可持久化、可恢复的 LangGraph 主图与领域子图，同时保留现有聊天、情绪、安全、画像、记忆、反馈和重新生成功能。

**Architecture:** FastAPI 只负责 HTTP/SSE 适配，由 `ConversationGraph` 路由到 `TurnGraph`、`RegenerationGraph` 和 `ProfileOnboardingGraph`。线程内状态写入官方 `AsyncSqliteSaver`，跨线程画像和长期记忆写入官方 `AsyncSqliteStore`；节点通过纯路由函数和小型领域仓储协作。

**Tech Stack:** Python 3.10+、FastAPI、LangChain、LangGraph 1.2.x、`langgraph-checkpoint-sqlite` 3.1.x、SQLite、aiosqlite、原生 HTML/CSS/JavaScript、pytest、pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-09-01-langgraph-architecture-design.md`

## Global Constraints

- Python 最低版本保持 3.10；异步节点显式接收 `Runtime` 和 `StreamWriter`，不依赖 Python 3.11+ 的隐式 contextvar 传播。
- 直接依赖固定为 `langgraph>=1.2,<1.3`、`langgraph-checkpoint-sqlite>=3.1,<4`、`aiosqlite>=0.20,<1`。
- 使用官方 `langgraph.store.sqlite.aio.AsyncSqliteStore` 实现规格中的 SQLite `BaseStore`，不自行重写通用 Store 协议。
- Checkpoint 文件固定为 `data/langgraph/checkpoints.sqlite3`，Store 文件固定为 `data/langgraph/store.sqlite3`；两者不得共用文件或私有表。
- Checkpointer 是线程消息、情绪状态和消息反馈的唯一事实源；Store 只保存 client-scoped 数据。
- 同一 `client_id` 的线程共享画像和长期记忆；不同线程隔离消息与情绪轨迹；不同 `client_id` 完全隔离。
- `LANGGRAPH_STRICT_MSGPACK=true`；Graph State 不使用 pickle fallback，不保存密钥、连接、锁、异常或流迭代器。
- SQLite 部署只支持单个 Uvicorn worker；不得把本实现描述为多进程生产方案。
- 旧运行数据库不读取、不迁移、不双写，也不在程序启动时自动删除。
- Benchmark、Ablation、情绪标签、Prompt variant 和报告格式保持兼容。
- 每个行为变更遵循红—绿—重构；每个任务只提交该任务列出的文件。
- 未经用户明确选择，不得启用 subagent 执行；默认可以使用 `superpowers:executing-plans` 内联执行。

---

## File Responsibility Map

### 新增文件

| 文件 | 单一职责 |
| --- | --- |
| `chatbot/models/graph.py` | 图共享的结构化值类型：风险、安全、情绪快照、线程元数据和请求结果 |
| `chatbot/models/events.py` | 稳定的图事件和 SSE DTO |
| `chatbot/graphs/state.py` | `ConversationState`、`ConversationInput`、`GraphContext` 和有界 reducer |
| `chatbot/graphs/routing.py` | 不产生副作用的 operation、情绪和安全路由函数 |
| `chatbot/graphs/dependencies.py` | `NodeDependencies` 及测试可替换的时钟/模型依赖 |
| `chatbot/graphs/nodes/input.py` | 输入校验、幂等检查和 HumanMessage 写入 |
| `chatbot/graphs/nodes/risk.py` | 风险预检和情绪分析触发 |
| `chatbot/graphs/nodes/emotion.py` | 情绪模型调用、轨迹更新和降级事件 |
| `chatbot/graphs/nodes/context.py` | client-scoped 画像和长期记忆加载 |
| `chatbot/graphs/nodes/generation.py` | Prompt 组装、普通/支持性/危机回复生成 |
| `chatbot/graphs/nodes/memory.py` | 单轮记忆抽取和周期性提炼副作用 |
| `chatbot/graphs/nodes/finalization.py` | 请求结果记录、状态裁剪和完成/重放事件 |
| `chatbot/graphs/turn.py` | 编译正常聊天子图 |
| `chatbot/graphs/regeneration.py` | 编译重新生成子图 |
| `chatbot/graphs/onboarding.py` | 编译画像草稿子图 |
| `chatbot/graphs/conversation.py` | 编译 operation 主路由图 |
| `chatbot/graphs/runtime.py` | 图调用门面、同线程锁、状态读取和简单状态更新 |
| `chatbot/persistence/runtime.py` | `AsyncSqliteSaver/Store` 生命周期和 Schema 初始化 |
| `chatbot/persistence/threads.py` | client/thread 归属与会话目录仓储 |
| `chatbot/memory/repository.py` | 基于 `BaseStore` 的异步长期记忆仓储 |
| `chatbot/memory/rules.py` | 从旧 SQLite provider 提取的纯记忆规则与排序函数 |
| `tests/graphs/...` | 节点、路由、子图和主图测试 |
| `tests/persistence/...` | 官方 SQLite Store/Saver 集成和进程重启测试 |

### 修改文件

| 文件 | 修改责任 |
| --- | --- |
| `requirements.txt` | 声明 LangGraph SQLite 和异步测试直接依赖 |
| `.env.example` | 增加新数据库路径和状态上限配置 |
| `chatbot/core/config.py` | 解析 `GraphConfig`，保持现有 LLM 配置兼容 |
| `chatbot/core/llm_adapter.py` | 增加 `ainvoke/astream` 协议与委托 |
| `chatbot/emotion/analysis.py` | 提供无持久化副作用的异步单次分析 |
| `chatbot/emotion/safety.py` | 增加上下文感知风险预检和确定性安全决策 |
| `chatbot/emotion/feedback.py` | 改为 client/thread-scoped Store 仓储 |
| `chatbot/profile/repository.py` | 改为 client-scoped Store 仓储 |
| `chatbot/web.py` | 使用 lifespan 和 Graph Runtime，提供新 API/SSE |
| `chatbot/static/index.html` | 增加会话新建/切换控件 |
| `chatbot/static/style.css` | 会话控件样式 |
| `chatbot/static/app.js` | bootstrap、多线程、流式 POST、反馈和重新生成协议 |
| `README.md` | 更新架构、依赖、配置、API、运行和限制 |
| `docs/README.md` | 将 LangGraph 设计和计划加入当前文档导航 |

### 最终删除文件

| 文件 | 删除条件 |
| --- | --- |
| `chatbot/chat_service.py` | 所有在线路径已通过 Graph Runtime 覆盖 |
| `chatbot/core/history.py` | 消息、反馈和重新生成测试已迁移到 Graph State |
| `chatbot/core/runtime_store.py` | 画像、情绪反馈和在线分析记录已迁移或取消 |
| `chatbot/core/llm.py` | Prompt 能力已迁入生成节点，进程内消息历史不再引用 |
| `chatbot/memory/sqlite.py` | 记忆规则与仓储测试已完整迁移 |
| `tests/app/test_chat_service.py` | 等价场景已在节点和图测试中覆盖 |
| `tests/core/test_history.py` | 等价场景已在重新生成和反馈测试中覆盖 |
| `tests/core/test_runtime_store.py` | 等价场景已在 Store 仓储测试中覆盖 |

---

### Task 1: 依赖、配置与图状态契约

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `chatbot/core/config.py`
- Create: `chatbot/models/__init__.py`
- Create: `chatbot/models/graph.py`
- Create: `chatbot/models/events.py`
- Create: `chatbot/graphs/__init__.py`
- Create: `chatbot/graphs/state.py`
- Create: `chatbot/graphs/routing.py`
- Test: `tests/core/test_config.py`
- Create: `tests/graphs/test_state.py`
- Create: `tests/graphs/test_routing.py`

**Interfaces:**
- Consumes: existing `ChatConfig` and LangChain `AnyMessage`/`add_messages`.
- Produces: `GraphConfig`, `ConversationState`, `ConversationInput`, `GraphContext`, `GraphEvent`, `RiskAssessment`, `SafetyDecision`, `EmotionSnapshot`, `RequestResult`, `should_analyze_emotion()` and route functions used by every later task.

- [ ] **Step 1: Add failing configuration tests**

```python
def test_load_graph_config_uses_local_sqlite_defaults(monkeypatch):
    for name in (
        "LANGGRAPH_CHECKPOINT_DB_PATH",
        "LANGGRAPH_STORE_DB_PATH",
        "LANGGRAPH_TIMELINE_LIMIT",
        "LANGGRAPH_REQUEST_HISTORY_LIMIT",
        "LANGGRAPH_STRICT_MSGPACK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLIENT_ID_SIGNING_SECRET", "test-signing-secret-with-at-least-32-bytes")

    config = load_graph_config()

    assert config.checkpoint_db_path == "data/langgraph/checkpoints.sqlite3"
    assert config.store_db_path == "data/langgraph/store.sqlite3"
    assert config.timeline_limit == 50
    assert config.request_history_limit == 64
    assert config.strict_msgpack is True


def test_load_graph_config_rejects_same_database_file(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB_PATH", "data/shared.sqlite3")
    monkeypatch.setenv("LANGGRAPH_STORE_DB_PATH", "data/shared.sqlite3")

    with pytest.raises(ConfigError, match="must use different files"):
        load_graph_config()
```

- [ ] **Step 2: Run the focused config tests and verify red**

Run: `.venv/bin/python -m pytest tests/core/test_config.py -k graph_config -v`

Expected: FAIL because `GraphConfig` and `load_graph_config` do not exist.

- [ ] **Step 3: Declare direct dependencies and install them**

Append these exact constraints to `requirements.txt`:

```text
langgraph>=1.2,<1.3
langgraph-checkpoint-sqlite>=3.1,<4
aiosqlite>=0.20,<1
pytest-asyncio>=0.25,<2
```

Add to `.env.example`:

```dotenv
LANGGRAPH_CHECKPOINT_DB_PATH=data/langgraph/checkpoints.sqlite3
LANGGRAPH_STORE_DB_PATH=data/langgraph/store.sqlite3
LANGGRAPH_TIMELINE_LIMIT=50
LANGGRAPH_REQUEST_HISTORY_LIMIT=64
LANGGRAPH_STRICT_MSGPACK=true
CLIENT_ID_SIGNING_SECRET=replace-with-at-least-32-random-bytes
```

Run: `.venv/bin/python -m pip install -r requirements.txt`

Expected: exit 0 and direct imports for `langgraph.checkpoint.sqlite.aio` and `langgraph.store.sqlite.aio` succeed.

- [ ] **Step 4: Implement `GraphConfig`**

Add this contract to `chatbot/core/config.py` and parse it with the existing `_clean`/`_parse_positive_int` style:

```python
@dataclass(frozen=True)
class GraphConfig:
    checkpoint_db_path: str
    store_db_path: str
    timeline_limit: int
    request_history_limit: int
    strict_msgpack: bool
    client_id_signing_secret: str


def load_graph_config() -> GraphConfig:
    checkpoint_path = _clean(os.getenv("LANGGRAPH_CHECKPOINT_DB_PATH")) or "data/langgraph/checkpoints.sqlite3"
    store_path = _clean(os.getenv("LANGGRAPH_STORE_DB_PATH")) or "data/langgraph/store.sqlite3"
    if Path(checkpoint_path).resolve() == Path(store_path).resolve():
        raise ConfigError("LangGraph checkpoint and store must use different files.")
    return GraphConfig(
        checkpoint_db_path=checkpoint_path,
        store_db_path=store_path,
        timeline_limit=_parse_positive_int_env("LANGGRAPH_TIMELINE_LIMIT", 50),
        request_history_limit=_parse_positive_int_env("LANGGRAPH_REQUEST_HISTORY_LIMIT", 64),
        strict_msgpack=_parse_required_true("LANGGRAPH_STRICT_MSGPACK", default=True),
        client_id_signing_secret=_required_secret("CLIENT_ID_SIGNING_SECRET", min_bytes=32),
    )
```

Import `Path`, add `_parse_positive_int_env(name, default)`, `_parse_required_true()` and `_required_secret()`, and test invalid/zero/short-secret values explicitly. Production startup must fail closed when the signing secret is missing or shorter than 32 UTF-8 bytes.

- [ ] **Step 5: Add failing state and routing tests**

```python
def test_capped_timeline_keeps_newest_items():
    assert capped_timeline(
        [{"turn_count": 1}, {"turn_count": 2}],
        [{"turn_count": 3}],
        limit=2,
    ) == [{"turn_count": 2}, {"turn_count": 3}]


@pytest.mark.parametrize(
    ("turn_count", "last_turn", "force", "expected"),
    [(1, 0, False, True), (3, 1, False, False), (6, 1, False, True), (3, 1, True, True)],
)
def test_should_analyze_emotion(turn_count, last_turn, force, expected):
    state = {
        "turn_count": turn_count,
        "last_emotion_analysis_turn": last_turn,
        "risk": {"signals": [], "force_emotion_analysis": force, "explicit_crisis": False},
    }
    assert should_analyze_emotion(state, emotion_interval=5) is expected


def test_route_by_safety_uses_dedicated_crisis_node():
    assert route_by_safety({"safety_state": {"level": "crisis"}}) == "generate_crisis_reply"
    assert route_by_safety({"safety_state": {"level": "supportive"}}) == "generate_reply"
```

- [ ] **Step 6: Run state/routing tests and verify red**

Run: `.venv/bin/python -m pytest tests/graphs/test_state.py tests/graphs/test_routing.py -v`

Expected: collection FAIL because graph model modules do not exist.

- [ ] **Step 7: Implement serializable models, state and pure routers**

Use `TypedDict`/`Literal`, not Pydantic model instances, inside Checkpoint values. The durable state contract must include:

```python
class ConversationState(MessagesState, total=False):
    turn_count: int
    emotion_state: dict[str, Any] | None
    emotion_timeline: list[EmotionSnapshot]
    recent_emotions: list[str]
    safety_state: SafetyDecision
    last_emotion_analysis_turn: int
    thread_meta: ThreadMetadata
    processed_requests: dict[str, RequestResult]
    operation: GraphOperation
    request_id: str
    input_message: str
    target_message_id: str
    regeneration_reason: str
    profile_answers: list[ProfileAnswer]
    risk: RiskAssessment
    profile_context: str
    memory_context: str
    response_content: str
    response_message_id: str
    error_code: str
```

`ThreadRecord` is a serializable frozen dataclass in `chatbot/models/graph.py` with exact fields `thread_id: str`, `title: str`, `created_at: str`, and `updated_at: str`. Store/checkpoint timestamps use timezone-aware UTC ISO-8601 strings, not `datetime` objects.

Define these exact pure-function signatures in `chatbot/graphs/routing.py`: `route_operation(state: ConversationState) -> Literal["turn", "regenerate", "onboard", "invalid"]`; `should_analyze_emotion(state: ConversationState, *, emotion_interval: int) -> bool`; `route_emotion(state: ConversationState, *, emotion_interval: int) -> Literal["analyze_emotion", "reuse_emotion"]`; and `route_by_safety(state: ConversationState) -> Literal["generate_reply", "generate_crisis_reply"]`.

`GraphEvent` uses `event: str` and `data: dict[str, Any]`; `GraphContext` uses `client_id`、`request_id`、`locale="zh-CN"`。

- [ ] **Step 8: Run Task 1 tests**

Run: `.venv/bin/python -m pytest tests/core/test_config.py tests/graphs/test_state.py tests/graphs/test_routing.py -v`

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add requirements.txt .env.example chatbot/core/config.py chatbot/models chatbot/graphs/__init__.py chatbot/graphs/state.py chatbot/graphs/routing.py tests/core/test_config.py tests/graphs
git commit -m "feat: add LangGraph state contracts"
```

---

### Task 2: 官方 SQLite Checkpointer/Store 生命周期与 client 仓储

**Files:**
- Create: `chatbot/persistence/__init__.py`
- Create: `chatbot/persistence/runtime.py`
- Create: `chatbot/persistence/threads.py`
- Modify: `chatbot/profile/repository.py`
- Modify: `chatbot/emotion/feedback.py`
- Create: `tests/persistence/test_runtime.py`
- Create: `tests/persistence/test_threads.py`
- Modify: `tests/profile/test_profile.py`
- Modify: `tests/emotion/test_emotion_feedback.py`

**Interfaces:**
- Consumes: `GraphConfig`, LangGraph `BaseStore`, `AsyncSqliteSaver`, `AsyncSqliteStore`, existing `sanitize_profile()`.
- Produces: `PersistenceHandles`, `open_persistence()`, `ThreadRepository`, async `load_profile()/save_profile()` and async emotion feedback functions.

- [ ] **Step 1: Write failing persistence lifecycle tests**

```python
@pytest.mark.asyncio
async def test_open_persistence_uses_separate_sqlite_files(tmp_path):
    config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
    )
    async with open_persistence(config) as handles:
        await handles.store.aput(("client-1", "profile"), "current", {"preferred_name": "小明"})
        item = await handles.store.aget(("client-1", "profile"), "current")
        assert item.value["preferred_name"] == "小明"

    assert (tmp_path / "checkpoints.sqlite3").exists()
    assert (tmp_path / "store.sqlite3").exists()
```

- [ ] **Step 2: Run the lifecycle test and verify red**

Run: `.venv/bin/python -m pytest tests/persistence/test_runtime.py -v`

Expected: FAIL because `open_persistence` is undefined.

- [ ] **Step 3: Implement the official persistence lifecycle**

Use the official adapters; do not implement `BaseStore.batch()` manually. Construct the Saver with the official serializer explicitly so strict mode does not depend on import-time environment ordering:

```python
@dataclass(frozen=True)
class PersistenceHandles:
    checkpointer: AsyncSqliteSaver
    store: AsyncSqliteStore


@asynccontextmanager
async def open_persistence(config: GraphConfig):
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
        store = await stack.enter_async_context(
            AsyncSqliteStore.from_conn_string(config.store_db_path)
        )
        await checkpointer.setup()
        await store.setup()
        yield PersistenceHandles(checkpointer=checkpointer, store=store)
```

`load_graph_config()` rejects `LANGGRAPH_STRICT_MSGPACK=false`; the explicit serializer is the enforcement mechanism. Add a test that an unsafe custom class fails serialization and leaves no Checkpoint row.

- [ ] **Step 4: Write failing client/thread repository tests**

```python
@pytest.mark.asyncio
async def test_thread_repository_isolates_clients(store):
    repo = ThreadRepository(store, now=lambda: FIXED_NOW)
    created = await repo.create("client-a", title="新对话")

    assert await repo.owns("client-a", created.thread_id) is True
    assert await repo.owns("client-b", created.thread_id) is False
    assert [item.thread_id for item in await repo.list("client-a")] == [created.thread_id]


@pytest.mark.asyncio
async def test_profile_is_shared_only_inside_client_namespace(store):
    await save_profile(store, "client-a", {"response_style": "简短"})
    assert await load_profile(store, "client-a") == {"response_style": "简短"}
    assert await load_profile(store, "client-b") == {}
```

- [ ] **Step 5: Run repository tests and verify red**

Run: `.venv/bin/python -m pytest tests/persistence/test_threads.py tests/profile/test_profile.py tests/emotion/test_emotion_feedback.py -v`

Expected: FAIL because repositories still use global `RuntimeStore`.

- [ ] **Step 6: Implement async namespaced repositories**

Define `ThreadRecord` in Task 1, then expose these exact methods: async `create(client_id, *, title="新对话") -> ThreadRecord`, async `list(client_id) -> list[ThreadRecord]`, async `owns(client_id, thread_id) -> bool`, async `touch(client_id, thread_id, *, title=None) -> None`, and async `delete_record(client_id, thread_id) -> None`.

Store thread keys under `(client_id, "threads")`; the key is `thread_id`. Sort newest `updated_at` first. `save_profile()` sanitizes before `aput((client_id, "profile"), "current", profile)`.

Emotion feedback validates the existing four feedback values, then writes a UUID-keyed record to `(client_id, "emotion_feedback", thread_id)` with message ID, turn count, predicted/corrected emotion and timestamp.

- [ ] **Step 7: Run Task 2 tests**

Run: `.venv/bin/python -m pytest tests/persistence tests/profile/test_profile.py tests/emotion/test_emotion_feedback.py -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add chatbot/persistence chatbot/profile/repository.py chatbot/emotion/feedback.py tests/persistence tests/profile/test_profile.py tests/emotion/test_emotion_feedback.py
git commit -m "feat: add LangGraph SQLite persistence"
```

---

### Task 3: 将长期记忆迁移到 BaseStore

**Files:**
- Create: `chatbot/memory/rules.py`
- Create: `chatbot/memory/repository.py`
- Modify: `chatbot/memory/models.py`
- Modify: `chatbot/memory/__init__.py`
- Rename: `tests/memory/test_local_memory.py` → `tests/memory/test_store_memory.py`
- Modify: `tests/memory/test_memory.py`
- Modify: `tests/memory/test_memory_consolidation.py`

**Interfaces:**
- Consumes: official `BaseStore`, existing `Memory`/`MemoryCandidate`, all conflict/ranking helpers currently in `chatbot/memory/sqlite.py`.
- Produces: `StoreMemoryRepository.asearch()`, `.aremember()`, `.aget_consolidation_state()` and `.amark_consolidated()` used by graph nodes.

- [ ] **Step 1: Rename the existing memory test suite and change only its fixture**

Use `git mv` so the existing duplicate/conflict/supersede/ranking cases remain reviewable:

```bash
git mv tests/memory/test_local_memory.py tests/memory/test_store_memory.py
```

Replace construction of `SQLiteLocalMemoryProvider(path)` with an async fixture that creates `AsyncSqliteStore`, calls `setup()`, and returns `StoreMemoryRepository(store)` plus `client_id="client-a"`. Do not delete assertions for negation, preference conflicts, superseding, recency or usage count.

- [ ] **Step 2: Add explicit namespace and idempotency tests**

```python
@pytest.mark.asyncio
async def test_memory_is_shared_across_threads_but_isolated_by_client(memory_repo):
    await memory_repo.aremember("client-a", [MemoryCandidate("用户喜欢简短回答。", "preference")])
    assert [m.content for m in await memory_repo.asearch("client-a", "简短", limit=5)] == ["用户喜欢简短回答。"]
    assert await memory_repo.asearch("client-b", "简短", limit=5) == []


@pytest.mark.asyncio
async def test_same_candidate_is_idempotent(memory_repo):
    candidate = MemoryCandidate("用户喜欢简短回答。", "preference")
    first = await memory_repo.aremember("client-a", [candidate])
    second = await memory_repo.aremember("client-a", [candidate])
    assert first[0].id == second[0].id
```

- [ ] **Step 3: Run the migrated suite and verify red**

Run: `.venv/bin/python -m pytest tests/memory/test_store_memory.py -v`

Expected: FAIL because `StoreMemoryRepository` does not exist.

- [ ] **Step 4: Extract pure rules without changing behavior**

Move normalization, topic detection, conflict detection, supersede decision, tokenization, lexical score, recency score and ranking score from `chatbot/memory/sqlite.py` into `chatbot/memory/rules.py`. Keep function inputs independent of SQLite rows.

Required exported functions are `normalize_content(value: str) -> str`, `deterministic_memory_key(value: str) -> str`, `is_similar_memory(existing: Memory, candidate: MemoryCandidate, content: str) -> bool`, `conflicts(existing: Memory, candidate: MemoryCandidate, content: str) -> bool`, `can_supersede(existing: Memory, candidate: MemoryCandidate) -> bool`, and `ranking_score(memory: Memory, query: str, query_tokens: set[str]) -> float`.

`deterministic_memory_key` returns `"mem_" + sha256(normalized UTF-8).hexdigest()`，使节点重试不会插入副本。

- [ ] **Step 5: Implement `StoreMemoryRepository`**

`StoreMemoryRepository` receives `store: BaseStore` and an injectable UTC clock. It exposes async `asearch(client_id, query, *, limit) -> list[Memory]`, `aremember(client_id, candidates) -> list[Memory]`, `aget_consolidation_state(client_id) -> dict[str, Any]`, and `amark_consolidated(client_id, *, turn_count, last_message_id, source_checkpoint_id) -> None`.

Use namespace `(client_id, "memories")`, store `status` and `supersedes_id` in each item value, and query only active items. Preserve existing maximum-confidence merge and last-used/use-count updates. Consolidation metadata uses `(client_id, "memory_meta")` key `"consolidation"` and records processed checkpoint IDs.

- [ ] **Step 6: Run all memory tests**

Run: `.venv/bin/python -m pytest tests/memory -v`

Expected: PASS with every migrated semantic test retained.

- [ ] **Step 7: Commit Task 3**

```bash
git add chatbot/memory tests/memory
git commit -m "feat: store long-term memory in LangGraph Store"
```

---

### Task 4: 异步模型接口、纯情绪分析与分层安全策略

**Files:**
- Modify: `chatbot/core/llm_adapter.py`
- Modify: `chatbot/emotion/analysis.py`
- Modify: `chatbot/emotion/safety.py`
- Modify: `chatbot/emotion/__init__.py`
- Modify: `tests/core/test_llm_adapter.py`
- Modify: `tests/emotion/test_emotion.py`
- Modify: `tests/emotion/test_safety.py`

**Interfaces:**
- Consumes: Task 1 `RiskAssessment`/`SafetyDecision`, existing emotion Prompt and parsers.
- Produces: async `ChatModelAdapter`, `analyze_emotion_async()`, `precheck_risk()` and `assess_safety()` used by TurnGraph nodes.

- [ ] **Step 1: Add failing async adapter and pure analysis tests**

```python
@pytest.mark.asyncio
async def test_adapter_delegates_ainvoke_to_client(monkeypatch):
    adapter = build_test_adapter(monkeypatch, async_content="ok")
    assert (await adapter.ainvoke("hello")).content == "ok"


@pytest.mark.asyncio
async def test_analyze_emotion_async_has_no_runtime_store_side_effect(monkeypatch):
    monkeypatch.setattr("chatbot.emotion.analysis.append_analysis_record", lambda record: pytest.fail("must not persist"))
    result = await analyze_emotion_async(
        AsyncFakeLlm(STRUCTURED_ANXIOUS_OUTPUT),
        [],
        "我有点担心明天。",
        turn_count=1,
        emotion_interval=5,
    )
    assert result.success is True
    assert result.state.primary_emotion == "anxious"
```

- [ ] **Step 2: Run focused async tests and verify red**

Run: `.venv/bin/python -m pytest tests/core/test_llm_adapter.py tests/emotion/test_emotion.py -k 'ainvoke or async_has_no' -v`

Expected: FAIL because async interfaces do not exist.

- [ ] **Step 3: Add `ainvoke/astream` and pure async analysis**

Update the adapter protocol and implementation:

```python
async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
    return await self._client.ainvoke(input, config=config, **kwargs)

async def astream(self, input: Any, config: Any = None, **kwargs: Any):
    async for chunk in self._client.astream(input, config=config, **kwargs):
        yield chunk
```

Implement `analyze_emotion_async` with the same prompt/parse semantics as the current function but no `RuntimeStore` calls. Keep the old sync wrapper temporarily so the still-unmigrated `ChatService` remains green until Task 13.

- [ ] **Step 4: Add failing contextual risk tests**

```python
@pytest.mark.parametrize(
    ("message", "force", "explicit"),
    [
        ("我今晚准备结束自己的生命。", True, True),
        ("我真的很绝望，感觉撑不住。", True, False),
        ("新闻里有人说想死。", True, False),
        ("我在准备明天的演示。", False, False),
    ],
)
def test_precheck_risk_distinguishes_context(message, force, explicit):
    result = precheck_risk(message)
    assert result["force_emotion_analysis"] is force
    assert result["explicit_crisis"] is explicit


def test_explicit_crisis_cannot_be_downgraded_by_normal_emotion():
    risk = {"signals": ["first_person_intent"], "force_emotion_analysis": True, "explicit_crisis": True}
    decision = assess_safety("我今晚准备结束自己的生命。", EmotionState("calm"), risk=risk)
    assert decision["level"] == "crisis"
```

- [ ] **Step 5: Run safety tests and verify red**

Run: `.venv/bin/python -m pytest tests/emotion/test_safety.py -v`

Expected: FAIL because `precheck_risk` and `risk=` are undefined.

- [ ] **Step 6: Implement risk precheck and safety decision**

Return stable reason codes, not free-form diagnostics. Export `precheck_risk(message: str) -> RiskAssessment` and extend `assess_safety(message: str, state: EmotionState | None, *, risk: RiskAssessment | None = None) -> SafetyDecision`.

Strong first-person intent/plan/action sets `explicit_crisis=True`; third-person, quotation and general discussion can set `force_emotion_analysis=True` but never `explicit_crisis=True` solely from the term. Preserve existing supportive keywords and confidence threshold.

- [ ] **Step 7: Run Task 4 tests and experiment regression tests**

Run: `.venv/bin/python -m pytest tests/core/test_llm_adapter.py tests/emotion tests/scripts/test_run_emotion_ablation.py -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add chatbot/core/llm_adapter.py chatbot/emotion tests/core/test_llm_adapter.py tests/emotion tests/scripts/test_run_emotion_ablation.py
git commit -m "feat: add async emotion and safety policies"
```

---

### Task 5: TurnGraph 的输入、风险、情绪与上下文节点

**Files:**
- Create: `chatbot/graphs/dependencies.py`
- Create: `chatbot/graphs/nodes/__init__.py`
- Create: `chatbot/graphs/nodes/input.py`
- Create: `chatbot/graphs/nodes/risk.py`
- Create: `chatbot/graphs/nodes/emotion.py`
- Create: `chatbot/graphs/nodes/context.py`
- Create: `tests/graphs/nodes/conftest.py`
- Create: `tests/graphs/nodes/test_input.py`
- Create: `tests/graphs/nodes/test_risk.py`
- Create: `tests/graphs/nodes/test_emotion.py`
- Create: `tests/graphs/nodes/test_context.py`

**Interfaces:**
- Consumes: Tasks 1–4 state types, routers, async emotion API, profile repository and `StoreMemoryRepository`.
- Produces: `NodeDependencies`, `accept_turn`, `risk_precheck`, `analyze_emotion_node`, `reuse_emotion`, `assess_safety_node` and `load_context`.

- [ ] **Step 1: Define fake dependencies shared by node tests**

```python
@dataclass
class FakeDependencies:
    chat_model: Any
    emotion_model: Any
    chat_config: ChatConfig
    graph_config: GraphConfig
    now: Callable[[], datetime] = lambda: FIXED_NOW


def make_runtime(client_id="client-a", request_id="req-1"):
    return SimpleNamespace(context=GraphContext(client_id, request_id, "zh-CN"), store=InMemoryStore())
```

Production `NodeDependencies` has the same five fields. Store access comes from `Runtime.store`, so fake and production nodes share signatures.

- [ ] **Step 2: Write failing input/idempotency tests**

```python
def test_accept_turn_adds_stable_human_message_and_increments_turn(writer, runtime):
    update = accept_turn(
        {"operation": "turn", "request_id": "req-1", "input_message": " 你好 ", "turn_count": 0},
        runtime,
        writer,
    )
    assert update["turn_count"] == 1
    assert update["messages"][0].id == "human_req-1"
    assert update["messages"][0].content == "你好"


def test_accept_turn_marks_completed_request_for_replay(writer, runtime):
    state = {
        "request_id": "req-1",
        "processed_requests": {"req-1": {"status": "completed", "message_id": "ai_req-1", "content": "已有回复"}},
    }
    assert accept_turn(state, runtime, writer)["replay_request"] is True
```

- [ ] **Step 3: Run input tests and verify red**

Run: `.venv/bin/python -m pytest tests/graphs/nodes/test_input.py -v`

Expected: FAIL because node modules do not exist.

- [ ] **Step 4: Implement input and risk nodes**

All node functions use `(state, runtime, writer)` parameters and return state deltas. `accept_turn` raises `GraphInputError("empty_message")` for blank input. It emits:

```python
writer({
    "event": "user_message",
    "data": {"message_id": human.id, "role": "human", "content": human.content},
})
```

`risk_precheck` calls `precheck_risk(state["input_message"])`, stores the result, and does not emit internal matched phrases to the browser.

- [ ] **Step 5: Write failing emotion node tests**

```python
@pytest.mark.asyncio
async def test_analyze_emotion_updates_timeline_and_emits_done(deps, runtime, writer):
    update = await analyze_emotion_node(
        {"messages": [HumanMessage(id="human_req-1", content="我很担心")], "turn_count": 1},
        runtime,
        writer,
        deps,
    )
    assert update["emotion_state"]["primary_emotion"] == "anxious"
    assert update["last_emotion_analysis_turn"] == 1
    assert writer.events[-1]["event"] == "emotion_done"


@pytest.mark.asyncio
async def test_emotion_failure_reuses_previous_state_and_sets_error(deps_with_failing_emotion, runtime, writer):
    update = await analyze_emotion_node(
        {"emotion_state": {"primary_emotion": "sad"}, "turn_count": 2, "messages": []},
        runtime,
        writer,
        deps_with_failing_emotion,
    )
    assert update["emotion_state"]["primary_emotion"] == "sad"
    assert update["error_code"] == "emotion_analysis_failed"
```

- [ ] **Step 6: Implement emotion/reuse/safety nodes**

Convert recent `HumanMessage/AIMessage` values to the existing analysis record shape. `analyze_emotion_node` catches model/parse failure and emits `emotion_error`; it must not write Store records. `reuse_emotion` only emits a state delta when no prior state exists. `assess_safety_node` always calls `assess_safety(..., risk=state["risk"])` after either emotion branch and emits a separate `safety` event.

- [ ] **Step 7: Write failing context isolation tests**

```python
@pytest.mark.asyncio
async def test_load_context_reads_profile_and_memory_for_runtime_client_only(deps, store, writer):
    await store.aput(("client-a", "profile"), "current", {"response_style": "简短"})
    await seed_memory(store, "client-a", "用户希望使用中文。")
    update = await load_context({"input_message": "继续", "recent_emotions": []}, make_runtime("client-a"), writer, deps)
    assert "简短" in update["profile_context"]
    assert "使用中文" in update["memory_context"]
```

- [ ] **Step 8: Implement context loading with explicit degradation**

`load_context` reads `runtime.context.client_id`, calls async profile and memory repositories, and returns empty context on Store read failure while emitting an internal warning through `logging`. It must not place raw Store `Item` objects into state.

- [ ] **Step 9: Run Task 5 tests**

Run: `.venv/bin/python -m pytest tests/graphs/nodes/test_input.py tests/graphs/nodes/test_risk.py tests/graphs/nodes/test_emotion.py tests/graphs/nodes/test_context.py -v`

Expected: PASS.

- [ ] **Step 10: Commit Task 5**

```bash
git add chatbot/graphs/dependencies.py chatbot/graphs/nodes tests/graphs/nodes
git commit -m "feat: add turn context and emotion nodes"
```

---

### Task 6: 回复生成、记忆副作用与完成节点

**Files:**
- Create: `chatbot/graphs/nodes/generation.py`
- Create: `chatbot/graphs/nodes/memory.py`
- Create: `chatbot/graphs/nodes/finalization.py`
- Create: `tests/graphs/nodes/test_generation.py`
- Create: `tests/graphs/nodes/test_memory.py`
- Create: `tests/graphs/nodes/test_finalization.py`
- Modify: `tests/core/test_llm.py`

**Interfaces:**
- Consumes: `NodeDependencies`, existing Prompt config, emotion/memory formatting, Store repositories.
- Produces: `build_chat_prompt()`, `generate_reply`, `generate_crisis_reply`, `extract_memory`, `maybe_consolidate`, `finalize_turn` and `replay_completed_request`.

- [ ] **Step 1: Port Prompt tests before moving implementation**

Move assertions from `tests/core/test_llm.py` for system message、画像、记忆和情绪上下文 into `tests/graphs/nodes/test_generation.py`:

```python
def test_build_chat_prompt_keeps_contexts_separate():
    prompt = build_chat_prompt(
        profile_context="User Profile:\n- response_style: 简短",
        memory_context="Relevant Long-term Memory:\n- 用户希望使用中文。",
        emotion_context="- primary: anxious",
        safety={"level": "supportive", "guidance": "先共情，再给下一步。"},
    )
    rendered = prompt.format_messages(input="继续", messages=[])
    assert "User Profile" in rendered[0].content
    assert "Relevant Long-term Memory" in rendered[0].content
    assert "primary: anxious" in rendered[0].content
```

- [ ] **Step 2: Run generation tests and verify red**

Run: `.venv/bin/python -m pytest tests/graphs/nodes/test_generation.py -v`

Expected: FAIL because generation module does not exist.

- [ ] **Step 3: Implement normal/supportive generation**

Use `ChatPromptTemplate` with a `MessagesPlaceholder("messages")` and call `await deps.chat_model.ainvoke(prompt_value, config=...)`. Return exactly one `AIMessage` with:

```python
AIMessage(
    id=f"ai_{state['request_id']}",
    content=content,
    additional_kwargs={
        "feedback": None,
        "turn_count": state["turn_count"],
        "emotion_state": state.get("emotion_state"),
        "predicted_emotion": primary_emotion,
        "safety_level": state["safety_state"]["level"],
    },
)
```

Do not manually iterate `astream`; LangGraph `messages` stream mode observes the model call. Return the message through the `messages` reducer and duplicate its ID/content into transient response fields for `finalize_turn`.

- [ ] **Step 4: Write and implement crisis buffering tests**

```python
@pytest.mark.asyncio
async def test_crisis_failure_returns_only_local_fallback(crisis_deps, runtime, writer):
    update = await generate_crisis_reply(CRISIS_STATE, runtime, writer, crisis_deps)
    assert "模型残片" not in update["response_content"]
    assert "可信任的人" in update["response_content"]
    assert update["messages"][0].additional_kwargs["safety_level"] == "crisis"
```

`generate_crisis_reply` calls `ainvoke`, validates non-empty final content, and only then returns it. On any failure it uses a locale-specific constant from `generation.py`; it never emits provider token deltas itself.

- [ ] **Step 5: Write failing memory side-effect tests**

```python
@pytest.mark.asyncio
async def test_extract_memory_is_idempotent_for_same_request(memory_deps, runtime, writer):
    state = completed_turn_state("我希望以后都用中文回答。", "好的。")
    await extract_memory(state, runtime, writer, memory_deps)
    await extract_memory(state, runtime, writer, memory_deps)
    memories = await memory_deps.memory_repository.asearch("client-a", "中文", limit=5)
    assert len(memories) == 1
```

- [ ] **Step 6: Implement memory extraction and consolidation nodes**

`extract_memory` calls existing `extract_memory_candidates`; `maybe_consolidate` uses existing `consolidation_due` and `extract_consolidated_memory_candidates`. Both catch Store failures, log with `request_id/thread_id`, and return a non-fatal `memory_warning` transient field. Both nodes accept `RunnableConfig`; `maybe_consolidate` reads `config["configurable"].get("checkpoint_id")` and uses `f"request:{request_id}"` as the deterministic pre-checkpoint fallback. Persist that exact value as `source_checkpoint_id` for consolidation idempotency.

- [ ] **Step 7: Write failing finalization and replay tests**

```python
def test_finalize_records_and_trims_processed_requests(graph_config, writer):
    state = state_with_65_completed_requests()
    update = finalize_turn(state, make_runtime(request_id="req-65"), writer, graph_config)
    assert len(update["processed_requests"]) == 64
    assert update["processed_requests"]["req-65"]["content"] == state["response_content"]
    assert update["input_message"] == ""


def test_replay_emits_existing_full_result(writer):
    replay_completed_request(REPLAY_STATE, make_runtime(), writer)
    assert writer.events[-1] == {
        "event": "done",
        "data": {"message_id": "ai_req-1", "content": "已有回复", "replayed": True},
    }
```

- [ ] **Step 8: Implement finalization and transient cleanup**

`finalize_turn` stores status `completed`, message ID and full content under the current `request_id`, trims oldest entries by completion time, updates `thread_meta.updated_at`, emits `done`, and clears operation-specific input, context, risk, response and error fields. `replay_completed_request` emits existing content without calling any model or Store write node.

- [ ] **Step 9: Run Task 6 tests**

Run: `.venv/bin/python -m pytest tests/graphs/nodes/test_generation.py tests/graphs/nodes/test_memory.py tests/graphs/nodes/test_finalization.py -v`

Expected: PASS.

- [ ] **Step 10: Commit Task 6**

```bash
git add chatbot/graphs/nodes/generation.py chatbot/graphs/nodes/memory.py chatbot/graphs/nodes/finalization.py tests/graphs/nodes tests/core/test_llm.py
git commit -m "feat: add generation and finalization nodes"
```

---

### Task 7: 编译并验证 TurnGraph

**Files:**
- Create: `chatbot/graphs/turn.py`
- Create: `tests/graphs/test_turn_graph.py`

**Interfaces:**
- Consumes: Tasks 5–6 nodes and Task 1 routers.
- Produces: `build_turn_graph(deps) -> StateGraph` and the tested main-turn execution path.

- [ ] **Step 1: Write failing happy-path graph test**

```python
@pytest.mark.asyncio
async def test_turn_graph_runs_first_turn_and_persists_messages(fake_deps):
    graph = build_turn_graph(fake_deps).compile(checkpointer=InMemorySaver(), store=InMemoryStore())
    config = {"configurable": {"thread_id": "thread-1"}}
    result = await graph.ainvoke(
        {"operation": "turn", "request_id": "req-1", "input_message": "我有点担心"},
        config,
        context=GraphContext("client-a", "req-1", "zh-CN"),
    )
    assert [message.type for message in result["messages"]] == ["human", "ai"]
    assert result["turn_count"] == 1
    assert result["last_emotion_analysis_turn"] == 1
```

- [ ] **Step 2: Add failing branch tests**

Cover these exact paths in separate tests:

```text
completed request -> replay_completed_request -> END
ordinary interval-not-due -> reuse_emotion
risk signal before interval -> analyze_emotion_node
normal/supportive -> generate_reply
explicit crisis -> generate_crisis_reply
emotion failure + explicit crisis -> generate_crisis_reply
memory failure -> reply still completed
```

- [ ] **Step 3: Run TurnGraph tests and verify red**

Run: `.venv/bin/python -m pytest tests/graphs/test_turn_graph.py -v`

Expected: FAIL because `build_turn_graph` is undefined.

- [ ] **Step 4: Build the graph with named nodes and conditional edges**

```python
def build_turn_graph(deps: NodeDependencies) -> StateGraph:
    builder = StateGraph(ConversationState, context_schema=GraphContext)
    builder.add_node("accept_turn", partial(accept_turn, deps=deps))
    builder.add_node("risk_precheck", partial(risk_precheck, deps=deps))
    builder.add_node("analyze_emotion", partial(analyze_emotion_node, deps=deps))
    builder.add_node("reuse_emotion", partial(reuse_emotion, deps=deps))
    builder.add_node("assess_safety", partial(assess_safety_node, deps=deps))
    builder.add_node("load_context", partial(load_context, deps=deps))
    builder.add_node("generate_reply", partial(generate_reply, deps=deps))
    builder.add_node("generate_crisis_reply", partial(generate_crisis_reply, deps=deps))
    builder.add_node("extract_memory", partial(extract_memory, deps=deps))
    builder.add_node("maybe_consolidate", partial(maybe_consolidate, deps=deps))
    builder.add_node("finalize_turn", partial(finalize_turn, deps=deps))
    builder.add_node("replay_completed_request", replay_completed_request)
    # Add the approved edges exactly as defined by the spec.
    return builder
```

The conditional edge after `accept_turn` checks `replay_request`; the emotion edge closes over `deps.chat_config.emotion_interval`.

- [ ] **Step 5: Verify custom and message streams**

Add one test using `astream(..., stream_mode=["messages", "custom"], version="v2")` and assert:

```python
assert "emotion_start" in custom_event_names
assert "emotion_done" in custom_event_names
assert "safety" in custom_event_names
assert "done" in custom_event_names
assert all(meta["langgraph_node"] == "generate_reply" for _, meta in message_parts)
```

- [ ] **Step 6: Run Task 7 tests**

Run: `.venv/bin/python -m pytest tests/graphs/test_turn_graph.py -v`

Expected: PASS.

- [ ] **Step 7: Commit Task 7**

```bash
git add chatbot/graphs/turn.py tests/graphs/test_turn_graph.py
git commit -m "feat: build LangGraph turn workflow"
```

---

### Task 8: 重新生成与画像引导子图

**Files:**
- Create: `chatbot/graphs/regeneration.py`
- Create: `chatbot/graphs/onboarding.py`
- Create: `tests/graphs/test_regeneration_graph.py`
- Create: `tests/graphs/test_onboarding_graph.py`
- Modify: `tests/profile/test_profile_onboarding.py`

**Interfaces:**
- Consumes: `ConversationState`, `NodeDependencies`, profile helpers and generation Prompt helpers.
- Produces: `build_regeneration_graph(deps)` and `build_onboarding_graph(deps)`.

- [ ] **Step 1: Write failing regeneration tests**

```python
@pytest.mark.asyncio
async def test_regeneration_replaces_same_message_id_and_preserves_original(fake_deps):
    graph = build_regeneration_graph(fake_deps).compile()
    old = AIMessage(id="ai_req-1", content="旧回复", additional_kwargs={"feedback": None})
    result = await graph.ainvoke({
        "operation": "regenerate",
        "request_id": "regen-1",
        "target_message_id": "ai_req-1",
        "regeneration_reason": "不准确",
        "messages": [HumanMessage(id="human_req-1", content="问题"), old],
        "turn_count": 1,
    })
    replacement = next(message for message in result["messages"] if message.id == "ai_req-1")
    assert replacement.content == "新回复"
    assert replacement.additional_kwargs["original_content"] == "旧回复"
    assert replacement.additional_kwargs["regenerated"] is True
```

Add separate errors for missing target, non-AI target, invalid reason and already-regenerated target.

- [ ] **Step 2: Run regeneration tests and verify red**

Run: `.venv/bin/python -m pytest tests/graphs/test_regeneration_graph.py -v`

Expected: FAIL because the graph does not exist.

- [ ] **Step 3: Implement regeneration nodes and graph**

The graph nodes are `validate_target -> rebuild_context -> load_context -> generate_variant -> replace_message -> finalize_regeneration`. Use the same target ID for replacement. Do not increment `turn_count`, alter current emotion, call memory extraction, or append a second AI message.

`finalize_regeneration` records request status and emits:

```python
writer({
    "event": "done",
    "data": {
        "message_id": target_id,
        "content": replacement.content,
        "reason": reason,
        "regenerated": True,
    },
})
```

- [ ] **Step 4: Write failing onboarding tests**

```python
@pytest.mark.asyncio
async def test_onboarding_returns_sanitized_model_draft(fake_deps):
    graph = build_onboarding_graph(fake_deps).compile()
    result = await graph.ainvoke({
        "operation": "onboard",
        "request_id": "profile-1",
        "profile_answers": [{"key": "response_style", "answer": " 简短 "}],
    })
    assert result["profile_draft"] == {"response_style": "简短"}


@pytest.mark.asyncio
async def test_onboarding_uses_fallback_when_model_fails(failing_deps):
    result = await build_onboarding_graph(failing_deps).compile().ainvoke(PROFILE_INPUT)
    assert result["profile_draft"] == {"response_style": "简短"}
```

- [ ] **Step 5: Implement onboarding graph without profile writes**

The graph nodes are `validate_answers -> draft_profile -> finalize_profile_draft`. Reuse `sanitize_profile` and `fallback_profile_draft`. The graph returns a draft only; `PUT /profile` remains the sole profile write path.

- [ ] **Step 6: Run Task 8 tests**

Run: `.venv/bin/python -m pytest tests/graphs/test_regeneration_graph.py tests/graphs/test_onboarding_graph.py tests/profile/test_profile_onboarding.py -v`

Expected: PASS.

- [ ] **Step 7: Commit Task 8**

```bash
git add chatbot/graphs/regeneration.py chatbot/graphs/onboarding.py tests/graphs/test_regeneration_graph.py tests/graphs/test_onboarding_graph.py tests/profile/test_profile_onboarding.py
git commit -m "feat: add regeneration and onboarding graphs"
```

---

### Task 9: ConversationGraph 与运行时门面

**Files:**
- Create: `chatbot/graphs/conversation.py`
- Create: `chatbot/graphs/runtime.py`
- Create: `tests/graphs/test_conversation_graph.py`
- Create: `tests/graphs/test_runtime.py`
- Modify: `chatbot/main.py`

**Interfaces:**
- Consumes: all three subgraphs, `PersistenceHandles`, Task 2 repositories and runtime LLM factory.
- Produces: `build_conversation_graph(deps)`, `ConversationRuntime` and `build_graph_runtime()` used exclusively by FastAPI.

- [ ] **Step 1: Write failing operation routing tests**

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["turn", "regenerate", "onboard"])
async def test_conversation_graph_routes_each_operation(operation, fake_deps):
    graph = build_conversation_graph(fake_deps).compile(checkpointer=InMemorySaver(), store=InMemoryStore())
    result = await graph.ainvoke(
        operation_input(operation),
        {"configurable": {"thread_id": "thread-1"}},
        context=GraphContext("client-a", "req-1", "zh-CN"),
    )
    assert result["operation"] == operation


@pytest.mark.asyncio
async def test_unknown_operation_returns_stable_error(fake_deps):
    with pytest.raises(GraphInputError, match="invalid_operation"):
        await compiled_graph(fake_deps).ainvoke({"operation": "unknown"})
```

- [ ] **Step 2: Run main graph tests and verify red**

Run: `.venv/bin/python -m pytest tests/graphs/test_conversation_graph.py -v`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement the lightweight parent graph**

Compile each child without its own checkpointer so the parent Checkpointer owns persistence:

```python
turn = build_turn_graph(deps).compile()
regeneration = build_regeneration_graph(deps).compile()
onboarding = build_onboarding_graph(deps).compile()

builder = StateGraph(ConversationState, context_schema=GraphContext)
builder.add_node("turn", turn)
builder.add_node("regenerate", regeneration)
builder.add_node("onboard", onboarding)
builder.add_node("invalid", reject_invalid_operation)
builder.add_conditional_edges(START, route_operation, {
    "turn": "turn",
    "regenerate": "regenerate",
    "onboard": "onboard",
    "invalid": "invalid",
})
```

- [ ] **Step 4: Write failing runtime serialization and restart tests**

```python
@pytest.mark.asyncio
async def test_runtime_serializes_same_thread_and_allows_other_thread(fake_runtime):
    lock_a1 = fake_runtime.lock_for("thread-a")
    lock_a2 = fake_runtime.lock_for("thread-a")
    lock_b = fake_runtime.lock_for("thread-b")
    assert lock_a1 is lock_a2
    assert lock_a1 is not lock_b


@pytest.mark.asyncio
async def test_runtime_restores_thread_after_persistence_reopen(tmp_path, fake_models):
    await run_one_turn_and_close(tmp_path, fake_models)
    state = await reopen_and_get_state(tmp_path, fake_models, "client-a", "thread-1")
    assert [m.content for m in state["messages"]] == ["你好", "回复"]
```

- [ ] **Step 5: Implement `ConversationRuntime`**

Required public methods are async `astream_turn(client_id, thread_id, request_id, message)`, async `astream_regeneration(client_id, thread_id, request_id, message_id, reason)`, async `ainvoke_profile_draft(client_id, thread_id, request_id, answers)`, async `aget_state(client_id, thread_id) -> StateSnapshot`, async `aupdate_message_feedback(client_id, thread_id, message_id, feedback) -> AIMessage`, async `adelete_thread(client_id, thread_id) -> None`, and synchronous `lock_for(thread_id) -> asyncio.Lock`.

Every public method verifies ownership via `ThreadRepository`. Turn/regeneration/feedback/delete acquire the per-thread lock. `build_graph_runtime(handles, config)` builds LLMs once, compiles the parent once with `checkpointer=handles.checkpointer` and `store=handles.store`, and returns the runtime.

Add this creation method so every Store thread record has a corresponding initial Checkpoint from the beginning:

```python
async def acreate_thread(self, client_id: str, *, title: str = "新对话") -> ThreadRecord:
    record = await self.thread_repository.create(client_id, title=title)
    try:
        await self.graph.aupdate_state(
            self._config(record.thread_id),
            {"thread_meta": {"title": record.title, "created_at": record.created_at, "updated_at": record.updated_at}},
        )
    except Exception:
        await self.thread_repository.delete_record(client_id, record.thread_id)
        raise
    return record
```

This prevents a newly-created but unused thread from being mistaken for a stale Store record during reconciliation.

- [ ] **Step 6: Implement safe thread deletion order**

`adelete_thread` calls `checkpointer.adelete_thread(thread_id)` first. Only after success does it delete the thread Store record. `ThreadRepository.list()` accepts an `exists` callback and removes stale records whose Checkpoint no longer exists. Saver failure leaves the Store record intact and raises `RuntimeOperationError("thread_delete_failed")`.

- [ ] **Step 7: Run Task 9 tests**

Run: `.venv/bin/python -m pytest tests/graphs/test_conversation_graph.py tests/graphs/test_runtime.py tests/persistence -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 9**

```bash
git add chatbot/graphs/conversation.py chatbot/graphs/runtime.py chatbot/main.py tests/graphs/test_conversation_graph.py tests/graphs/test_runtime.py tests/persistence
git commit -m "feat: add persistent conversation graph runtime"
```

---

### Task 10: FastAPI lifespan、client/thread API 与非流式领域端点

**Files:**
- Modify: `chatbot/web.py`
- Create: `tests/app/test_graph_api.py`
- Modify: `tests/app/test_web.py`

**Interfaces:**
- Consumes: `open_persistence()`, `build_graph_runtime()`, `ConversationRuntime`, Store profile/emotion repositories and existing Pydantic compatibility helpers.
- Produces: application lifespan, anonymous client bootstrap, thread CRUD/read endpoints, profile endpoints, message feedback and emotion feedback endpoints.

- [ ] **Step 1: Write failing lifespan and bootstrap tests**

```python
def test_bootstrap_returns_signed_client_and_first_thread(app_client):
    response = app_client.post("/api/clients/bootstrap")
    assert response.status_code == 201
    payload = response.json()
    assert payload["client_id"].startswith("c_")
    assert payload["thread"]["thread_id"]


def test_bootstrap_client_token_rejects_tampering(app_client):
    payload = app_client.post("/api/clients/bootstrap").json()
    bad = payload["client_id"][:-1] + ("a" if payload["client_id"][-1] != "a" else "b")
    assert app_client.get(f"/api/clients/{bad}/threads").status_code == 401
```

Use a test-only signing secret and temporary graph database paths. The fixture must enter FastAPI lifespan so SQLite handles open and close in the test.

- [ ] **Step 2: Run bootstrap tests and verify red**

Run: `.venv/bin/python -m pytest tests/app/test_graph_api.py -k bootstrap -v`

Expected: FAIL because the client bootstrap and lifespan runtime do not exist.

- [ ] **Step 3: Replace startup hooks with lifespan and signed anonymous identity**

Use `@asynccontextmanager` and construct the runtime once:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_strict_msgpack()
    graph_config = load_graph_config()
    async with open_persistence(graph_config) as handles:
        app.state.graph_runtime = build_graph_runtime(handles, graph_config)
        yield
```

Generate 32 random bytes for the opaque identity payload and authenticate it with HMAC-SHA256 using `CLIENT_ID_SIGNING_SECRET`. Use URL-safe base64 without exposing the secret. Invalid signatures return `401 invalid_client_id`; a valid client with a thread owned by another client receives `404 thread_not_found`, avoiding an ownership oracle.

- [ ] **Step 4: Write failing thread ownership and restart tests**

Cover these endpoints and exact outcomes:

```text
GET    /api/clients/{client_id}/threads                         -> newest first
POST   /api/clients/{client_id}/threads                         -> 201 + initialized checkpoint
GET    /api/clients/{client_id}/threads/{thread_id}             -> messages + latest emotion + metadata
DELETE /api/clients/{client_id}/threads/{thread_id}             -> 204
```

Tests must assert cross-client read/delete returns `404`, delete removes Saver state before Store record, and closing/reopening the app with the same temporary DBs restores the thread and messages.

- [ ] **Step 5: Implement thread endpoints through `ConversationRuntime` only**

Do not query Saver private tables. Serialize `StateSnapshot.values` into the existing browser message DTO, including feedback, regeneration metadata, safety level, predicted emotion, and the current emotion snapshot. If a thread record lacks a checkpoint during listing, call the Task 9 reconciliation path and omit it.

- [ ] **Step 6: Write failing profile and feedback endpoint tests**

Add four explicit tests: profile values are visible from two threads owned by one client but absent for another client; profile draft rejects a missing or unowned `thread_id`; message feedback returns the same AI message ID with the new value; emotion feedback can be listed only from the owning client/thread namespace.

The API surface is:

```text
GET   /api/clients/{client_id}/profile
PUT   /api/clients/{client_id}/profile
POST  /api/clients/{client_id}/profile/draft
PATCH /api/clients/{client_id}/threads/{thread_id}/messages/{message_id}/feedback
POST  /api/clients/{client_id}/threads/{thread_id}/emotion-feedback
GET   /api/clients/{client_id}/threads/{thread_id}/emotion-timeline
```

`profile/draft` includes `thread_id` in the request body and invokes graph operation `onboard`; only `PUT /profile` persists a profile.

- [ ] **Step 7: Implement profile, timeline and feedback endpoints**

Map stable domain errors to HTTP status codes in one helper: malformed input `422`, invalid identity `401`, missing/unowned thread or message `404`, already-rated/conflict `409`, and persistence/model failure `500`. Message feedback calls `ConversationRuntime.aupdate_message_feedback()` so the Checkpoint remains the source of truth. Emotion feedback writes the Task 2 Store repository.

- [ ] **Step 8: Remove superseded non-streaming routes and run Task 10 tests**

Remove `/api/history`, `/api/session`, global `/api/profile`, `/api/emotion/timeline`, `/api/messages/{id}/feedback`, `/api/emotion/feedback`, and the old non-streaming regeneration POST after their replacement tests are green.

Run: `.venv/bin/python -m pytest tests/app/test_graph_api.py tests/app/test_web.py -k 'not stream and not static_app_js' -v`

Expected: PASS.

- [ ] **Step 9: Commit Task 10**

```bash
git add chatbot/web.py tests/app/test_graph_api.py tests/app/test_web.py
git commit -m "feat: expose client-scoped graph APIs"
```

---

### Task 11: POST 流式适配、稳定 SSE 事件与失败语义

**Files:**
- Modify: `chatbot/web.py`
- Create: `tests/app/test_graph_streaming.py`
- Modify: `tests/app/test_web.py`

**Interfaces:**
- Consumes: `ConversationRuntime.astream_turn()`, `astream_regeneration()` and LangGraph v2 stream chunks in `messages`/`custom` modes.
- Produces: POST response streams for normal turns and regeneration with the approved stable browser event contract.

- [ ] **Step 1: Write failing stable event adapter tests**

```python
@pytest.mark.asyncio
async def test_event_adapter_filters_non_generation_message_chunks():
    chunks = fake_v2_chunks(
        custom=[("emotion_start", {}), ("emotion_done", {"primary_emotion": "anxious"}), ("safety", {"level": "supportive"})],
        messages=[("分析内部文本", {"langgraph_node": "analyze_emotion"}), ("你好", {"langgraph_node": "generate_reply"})],
    )
    events = [event async for event in adapt_graph_stream(chunks)]
    assert [event.event for event in events] == ["emotion_start", "emotion_done", "safety", "token"]
    assert events[-1].data["content"] == "你好"
```

Add tests for every public event name: `run_started`, `user_message`, `emotion_start`, `emotion_done`, `emotion_error`, `safety`, `token`, `done`, `error`. Assert `emotion_done` never carries the final safety decision.

- [ ] **Step 2: Run adapter tests and verify red**

Run: `.venv/bin/python -m pytest tests/app/test_graph_streaming.py -k adapter -v`

Expected: FAIL because `adapt_graph_stream` does not exist.

- [ ] **Step 3: Implement the LangGraph-to-SSE adapter**

Subscribe with `stream_mode=["messages", "custom"]` and `version="v2"`. Pass custom events through the typed `GraphEvent` allowlist. Convert model chunks to `token` only when `metadata["langgraph_node"]` is `generate_reply` or `generate_variant`; crisis generation is buffered by its node and is emitted once from its completion event. Encode with the existing `_format_sse()` and send periodic comment heartbeats without changing the public event contract.

- [ ] **Step 4: Write failing POST streaming integration tests**

```python
def test_turn_stream_uses_post_body_and_has_no_stream_id_registry(seed_thread):
    response = client.post(
        f"/api/clients/{client_id}/threads/{thread_id}/messages:stream",
        json={"message": "你好", "request_id": "req-1"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert event_names(response.text)[0] == "run_started"
    assert event_names(response.text)[-1] == "done"
```

Add separate assertions that resending a completed request leaves one HumanMessage and emits `replayed: true`; a normal-generation exception leaves zero partial AIMessages and permits retry with the same request ID; and regeneration `done.message_id` equals the target message ID while the persisted thread still contains only one message with that ID.

The two endpoints are:

```text
POST /api/clients/{client_id}/threads/{thread_id}/messages:stream
POST /api/clients/{client_id}/threads/{thread_id}/messages/{message_id}/regenerate:stream
```

- [ ] **Step 5: Implement streaming endpoints and cancellation handling**

Validate identity/ownership before constructing `StreamingResponse`. Turn input contains client-generated UUID `request_id`; regeneration contains `request_id` and one approved reason. On disconnect, cancel stream consumption and release the per-thread lock in `finally`. Domain errors emitted after headers become `error` events; validation and ownership errors discovered before headers remain ordinary JSON HTTP errors.

For a normal model failure, do not append an incomplete `AIMessage`; retrying the same non-completed `request_id` reuses the existing stable HumanMessage ID. For a completed `request_id`, emit the stored `done` payload with `replayed: true` and do not call models or memory nodes.

- [ ] **Step 6: Delete the one-time stream registry and old GET stream route**

Remove `POST /api/chat/streams`, `GET /api/chat/streams/{stream_id}`, the in-memory stream request map, one-time lookup helpers, and their obsolete tests. Retain `_format_sse()` if the new adapter uses it.

- [ ] **Step 7: Run Task 11 tests**

Run: `.venv/bin/python -m pytest tests/app/test_graph_streaming.py tests/app/test_web.py -k 'stream or format_sse' -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 11**

```bash
git add chatbot/web.py tests/app/test_graph_streaming.py tests/app/test_web.py
git commit -m "feat: stream LangGraph events over post SSE"
```

---

### Task 12: 浏览器 client/thread 状态与 Fetch ReadableStream UI

**Files:**
- Modify: `chatbot/static/index.html`
- Modify: `chatbot/static/style.css`
- Modify: `chatbot/static/app.js`
- Modify: `tests/app/test_web.py`

**Interfaces:**
- Consumes: Tasks 10–11 JSON/SSE APIs.
- Produces: persistent anonymous browser identity, thread creation/switching/deletion, streamed chat/regeneration, profile and feedback UI without `EventSource`.

- [ ] **Step 1: Add failing DOM tests for bootstrap and thread controls**

Extend the existing Node-based browser harness in `tests/app/test_web.py`:

```javascript
assert.equal(localStorage.getItem("hdu_erc_client_id"), "c_signed");
assert.equal(localStorage.getItem("hdu_erc_thread_id"), "thread-1");
assert.equal(document.querySelectorAll("[data-thread-id]").length, 2);
```

Cover first launch bootstrap, reload reuse, invalid stored client re-bootstrap, new thread, switch thread, delete current thread, and automatic creation when the last thread is removed.

- [ ] **Step 2: Run browser state tests and verify red**

Run: `.venv/bin/python -m pytest tests/app/test_web.py -k 'bootstrap or thread_control' -v`

Expected: FAIL because the DOM and client-scoped bootstrap logic do not exist.

- [ ] **Step 3: Add thread markup/style and bootstrap controller**

Add accessible controls with stable IDs: `#thread-list`, `#new-thread-button`, and `#delete-thread-button`. Store only the signed opaque `client_id` and current `thread_id` in `localStorage`; messages, profiles, feedback and emotions stay server-side. All client-scoped URLs use `encodeURIComponent` for both IDs. Switching threads aborts any current stream and reloads the selected snapshot and timeline.

- [ ] **Step 4: Add failing incremental SSE parser tests**

Test chunk splits inside UTF-8 text, JSON, CRLF, and between the two newline frame terminators:

```javascript
const frames = await collectSseFrames(["event: tok", "en\ndata: {\"content\":\"你", "好\"}\n\n"]);
assert.deepEqual(frames, [{event: "token", data: {content: "你好"}}]);
```

Also assert `EventSource` is never constructed and POST request bodies contain `message`/`request_id` or `reason`/`request_id`.

- [ ] **Step 5: Implement `fetch()` + `ReadableStream` streaming**

Use `TextDecoder("utf-8", {stream: true})`, retain an incomplete-frame buffer, normalize CRLF, split only on a blank line, join multiple `data:` lines, then JSON-decode. Keep one `AbortController` per active request. Map stable events to existing status/message UI; `safety` alone controls crisis/supportive presentation.

- [ ] **Step 6: Port feedback, regeneration, profile and timeline requests**

Replace global endpoints with client/thread URLs. Regeneration consumes its stream and replaces the existing message DOM node with the same message ID; it does not insert a second assistant bubble. Preserve `original_content`, `regenerated`, reason, feedback controls and predicted emotion metadata from thread snapshots.

- [ ] **Step 7: Run all static UI tests**

Run: `.venv/bin/python -m pytest tests/app/test_web.py -k 'static or index or feedback or regenerat or thread or bootstrap' -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 12**

```bash
git add chatbot/static/index.html chatbot/static/style.css chatbot/static/app.js tests/app/test_web.py
git commit -m "feat: add multi-thread streaming chat UI"
```

---

### Task 13: 移除旧在线架构并迁移剩余回归测试

**Files:**
- Delete: `chatbot/chat_service.py`
- Delete: `chatbot/core/history.py`
- Delete: `chatbot/core/runtime_store.py`
- Delete: `chatbot/core/llm.py`
- Delete: `chatbot/memory/sqlite.py`
- Delete: `tests/app/test_chat_service.py`
- Delete: `tests/core/test_history.py`
- Delete: `tests/core/test_runtime_store.py`
- Modify: `chatbot/emotion/analysis.py`
- Modify: `tests/emotion/test_emotion.py`
- Modify: `tests/core/test_llm.py`
- Modify: `tests/memory/test_local_memory.py`
- Modify: `tests/app/test_web.py`

**Interfaces:**
- Consumes: green graph, repository and API tests from Tasks 1–12.
- Produces: one online execution path with no `ChatService`, process-local history, global RuntimeStore, or bespoke SQLite memory provider.

- [ ] **Step 1: Build a legacy behavior coverage ledger before deletion**

For every test in the three files marked for deletion, record its replacement test node in a table inside the Task 13 commit message draft. Required categories are: bounded history, role filtering, exact message replacement, feedback conflict, regeneration metadata, memory score ordering, consolidation idempotency, profile persistence, emotion feedback and online failure degradation. Do not delete a legacy test until its category has a green replacement.

- [ ] **Step 2: Move remaining valuable tests to their new owners**

Use `git mv tests/memory/test_local_memory.py tests/memory/test_store_memory.py`, then rewrite its fixtures for `InMemoryStore`/`AsyncSqliteStore` while preserving ranking, scope, retention, consolidation and idempotency assertions. Move Prompt assertions from `tests/core/test_llm.py` to `tests/graphs/nodes/test_generation.py`. Retain emotion label, prompt, parser and fallback assertions in `tests/emotion/test_emotion.py`; remove only RuntimeStore persistence assertions already covered by graph timeline/checkpoint tests.

- [ ] **Step 3: Run migrated tests before deleting implementation**

Run: `.venv/bin/python -m pytest tests/memory/test_store_memory.py tests/graphs tests/persistence tests/emotion tests/profile tests/app/test_graph_api.py tests/app/test_graph_streaming.py -v`

Expected: PASS.

- [ ] **Step 4: Delete the old implementation and obsolete test files**

Delete only the listed files. Remove imports and compatibility wrappers from `chatbot/emotion/analysis.py`, `chatbot/main.py`, package `__init__.py` files, tests and web wiring. The offline ablation script must continue constructing prompt/model/parser output directly and must not depend on Graph Checkpoints.

- [ ] **Step 5: Prove the online path has no legacy references**

Run:

```bash
rg -n 'ChatService|RunnableWithMessageHistory|InMemoryChatMessageHistory|get_session_history|RuntimeStore|SQLiteLocalMemoryProvider|/api/chat/streams' chatbot tests README.md
```

Expected: no matches. Historical design/spec documents are excluded from this assertion.

- [ ] **Step 6: Run complete tests after deletion**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 13**

```bash
git add -A chatbot tests
git commit -m "refactor: remove legacy chat runtime"
```

---

### Task 14: 文档、部署限制与端到端验收

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Create: `tests/app/test_langgraph_acceptance.py`

**Interfaces:**
- Consumes: final runtime/API/UI behavior.
- Produces: reproducible local quickstart, exact API map, explicit SQLite limits and executable acceptance coverage.

- [ ] **Step 1: Write failing acceptance tests from the approved scenarios**

Use deterministic fake chat/emotion models and temporary real SQLite files. Cover:

```text
bootstrap -> first thread -> first turn -> emotion analysis -> done
same client -> second thread -> profile/memory shared; messages/emotion isolated
different client -> no profile/memory/thread access
ordinary interval turn -> emotion reused
risk signal -> forced analysis
explicit first-person intent -> crisis cannot be downgraded
like/dislike -> same AIMessage updated
regenerate with reason -> same message ID replaced
close/reopen app -> state, feedback and thread list restored
delete thread -> checkpoint then directory record removed
Store write failure -> reply still completes with warning
normal generation failure -> no partial AIMessage; same request can retry
```

- [ ] **Step 2: Run acceptance tests and verify red**

Run: `.venv/bin/python -m pytest tests/app/test_langgraph_acceptance.py -v`

Expected: FAIL for any remaining integration gap.

- [ ] **Step 3: Fix only integration gaps and make acceptance green**

Keep fixes in the owning modules. Do not add a second orchestration layer in `web.py`. Re-run each failing focused test before continuing.

- [ ] **Step 4: Update the Chinese quickstart and architecture documentation**

README must include:

- Python 3.10+ install and environment commands;
- required LLM variables and `CLIENT_ID_SIGNING_SECRET` generation guidance;
- both LangGraph SQLite paths and `LANGGRAPH_STRICT_MSGPACK=true`;
- one Uvicorn worker command and an explicit statement that local SQLite is not a multi-process production deployment;
- parent/subgraph structure and Checkpointer-versus-Store ownership;
- complete client/thread/profile/feedback/stream endpoint table;
- old databases are ignored, not migrated or automatically deleted;
- browser identity reset behavior and data-loss implication;
- benchmark/ablation commands and their retained compatibility.

Add links to the spec and this implementation plan in `docs/README.md`.

- [ ] **Step 5: Run the fresh-environment verification suite**

Using Python 3.12:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q chatbot scripts tests
git diff --check
```

Expected: all tests pass, compileall exits 0, and `git diff --check` has no output.

- [ ] **Step 6: Perform browser smoke QA when credentials are available**

Start exactly one Uvicorn worker against fresh temporary graph databases. Verify create/switch/delete thread, normal streaming, emotion timeline, feedback, regeneration and reload recovery in a real browser. If valid LLM credentials are unavailable, record that only deterministic fake-model integration was executed; do not call it a live LLM verification.

- [ ] **Step 7: Inspect final scope and commit documentation/acceptance**

Run:

```bash
git status --short
git diff --stat main...HEAD
git log --oneline --decorate main..HEAD
```

Confirm no secrets, `.env`, SQLite database, `.venv`, generated reports or unrelated user files are staged.

```bash
git add README.md docs/README.md tests/app/test_langgraph_acceptance.py
git commit -m "docs: document LangGraph conversation runtime"
```

---

## Spec Coverage Matrix

| Approved requirement | Implemented/tested by |
| --- | --- |
| Parent graph + Turn/Regeneration/Onboarding subgraphs | Tasks 7–9 |
| Checkpointer as thread source of truth | Tasks 1–2, 7, 9–11 |
| client-scoped shared profile and memory | Tasks 2–3, 5, 10, 14 |
| thread-scoped messages/emotion/feedback | Tasks 1–2, 5–7, 9–10 |
| anonymous browser client and multiple threads | Tasks 9–10, 12 |
| first/interval/risk-triggered emotion analysis | Tasks 1, 4–5, 7, 14 |
| deterministic contextual crisis routing | Tasks 4–7, 14 |
| normal/supportive/crisis failure semantics | Tasks 5–7, 11, 14 |
| streaming chat with stable events and no stream ID map | Tasks 7, 11–12 |
| likes/dislikes and reason-based same-ID regeneration | Tasks 8–12, 14 |
| profile onboarding draft with explicit save | Tasks 8, 10, 12 |
| two local SQLite files, strict serialization, one worker | Tasks 1–2, 9, 14 |
| restart recovery, idempotency and bounded state | Tasks 1–3, 6–7, 9, 11, 14 |
| no old-data migration or automatic deletion | Tasks 13–14 |
| benchmark/ablation compatibility | Tasks 3–4, 13–14 |
| removal of legacy online orchestration | Task 13 |

## Execution Handoff

Execute Tasks 1–14 in order. At every checkpoint, keep the listed focused tests green and commit only the files owned by that task. Do not merge, push, deploy, delete the worktree, or claim real-LLM validation without separate authorization and evidence.
