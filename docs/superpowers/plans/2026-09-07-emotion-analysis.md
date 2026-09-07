# Per-turn Emotion Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本项目禁止未经用户请求使用子代理，按当前会话串行执行。

**Goal:** 每条新输入在回复前执行可审计的情绪分析，使用 34 个可配置选项，并在 60% token 预算内保留最近的完整对话轮次。

**Architecture:** 在现有主图 prepare_turn 和 generate_response 之间接入独立情绪子图。业务输入、配置快照、调用输出与失败信息落入独立 SQLite 表；图状态只携带记录 ID 和控制状态。识别成功将本轮结果加入回复上下文，模型或解析失败落库后继续普通回复。

**Tech Stack:** Python、FastAPI、LangGraph 1.2.11、langgraph-checkpoint-sqlite 3.1.1、langchain-openai 1.6.0、aiosqlite、pytest/pytest-asyncio、JSON。计数使用已安装 LangChain 模型计数接口，并校验其支持的模型格式，不新增无必要依赖。

**Spec:** [2026-09-07-emotion-analysis-design.md](../specs/2026-09-07-emotion-analysis-design.md)

## Global Constraints

- 沿用归档 v1 的 32 类情绪并增加 neutral（中性）和 no_emotion（无情绪），共 34 个识别选项。
- 每条新消息分析一次，不设置轮次间隔，不复用上轮情绪作为本轮判断。
- 聊天内容（含当前输入和消息封装开销）最多占情绪模型上下文上限的 60%，按 token 计算。
- 只按整轮移除最早历史，不摘要或截断单条消息；只取当前 conversation_id 且截至当前输入的历史。
- 相同 request_id 复用已存在分析；不同 request_id 即使文本相同也重新分析。
- 分析失败保存可获取事实后继续普通回复；数据库写入失败走基础设施失败路径。
- 配置启动时校验，重启生效；保存实际标签、family、示例、提示词、参数快照和计数信息。
- no_emotion 作为主选项时 secondary_emotions 必须为空，且不能作为次要选项；失败不能伪装为 neutral 或 no_emotion。
- 情绪模型不自动重试，适配器 max_retries=0；本期不引入 attempt 多次调用表。
- 不迁入旧版记忆系统、安全路由或前端改版；不改聊天回复的原有历史条数策略。
- 当前 worktree：/Users/oriki/Database/HDU-erc/.worktrees/emotion-analysis-20260907；任务分支 codex/emotion-analysis-20260907；目标 main。
- 所有写入、测试和提交均在此 worktree 执行，禁止复制原目录的未提交文件。若实现确实依赖原目录未提交内容，停止写入并询问用户。
- 不自动合并、推送、删除分支或 worktree。每个任务验证后仅提交其自身文件。

## 执行环境与基线

所有下述命令从本 worktree 根目录执行。可复用原目录虚拟环境中的解释器，但 PYTHONPATH 和工作目录必须指向本 worktree；不从原目录导入 chatbot，不加载其 .env，不使用其数据库。

```bash
git status --short
git branch --show-current
/Users/oriki/Database/HDU-erc/.venv/bin/python -c 'import chatbot; print(chatbot.__file__)'
/Users/oriki/Database/HDU-erc/.venv/bin/python -m pytest -q
```

预期分支正确、工作区干净、模块路径位于本 worktree，离线基线通过。真实模型测试遵循现有环境开关，不为此计划访问真实 API。基线若失败先确认原因，不能将现有问题混入功能提交。下面的 `python` 均指上面的虚拟环境解释器。

## 文件与职责

| 文件 | 职责 |
|---|---|
| config/emotion_labels.json、config/emotion_families.json | 34 个选项的定义与归类 |
| config/emotion_examples.json | 从归档迁入的示例及 neutral/no_emotion 示例 |
| chatbot/emotion/types.py | 配置、预算、结果、调用与审计的数据契约 |
| chatbot/emotion/config.py | 文件解析、重复键检查、模型设置与计数设置 |
| chatbot/emotion/history.py、budget.py | 完整轮次配对、历史选择和预算复核 |
| chatbot/emotion/prompt.py、retrieval.py、parsing.py | few-shot 检索、角色化提示词、结构化输出校验 |
| chatbot/emotion/model.py | 独立模型调用及安全诊断边界 |
| chatbot/emotion/graph.py | 准备、调用、结果落库的子图 |
| chatbot/db/emotions.py | emotion_analyses 仓储和持久化状态转换 |
| chatbot/db/schema.py、messages.py | v1→v2 迁移、按消息边界取历史 |
| chatbot/graph/{state,dependencies,nodes,builder}.py | 主图接入，按记录 ID 获取本轮结果 |
| chatbot/llm/prompt.py | 可选的本轮情绪上下文 |
| chatbot/web.py、services/recovery.py | 应用装配与启动中断恢复 |
| tests/emotion/、tests/db/test_emotions.py | 情绪模块单测和真实 SQLite 测试 |
| tests/graph/test_emotion_turn.py、tests/app/test_emotion_flow.py | 图执行和应用回归 |
| README.md、.env.example | 配置、查询审计、异常语义与运行说明 |

除少量必要接入外不重构现有大型 TurnNodes。新失败处理集中于情绪模块；原图的 terminal、幂等、stream flush 与清理约束必须保留。

## Task 1: 定义标签配置及共享数据契约

**Files:** Create `chatbot/emotion/__init__.py`, `types.py`, `config.py`, `config/emotion_labels.json`, `config/emotion_families.json`, `tests/emotion/test_config.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class Taxonomy:
    labels: dict[str, str]
    families: dict[str, str]
    content_hash: str

@dataclass(frozen=True)
class BudgetConfig:
    context_tokens: int
    output_tokens: int
    safety_tokens: int
    history_ratio: float = 0.60

class TokenCounter(Protocol):
    identity: str
    version: str
    def count(self, messages: Sequence[BaseMessage]) -> int: ...

@dataclass(frozen=True)
class EmotionResult:
    primary_emotion: str
    confidence: float
    secondary_emotions: list[str]
    evidence: str
    reply_strategy: str
    trajectory_note: str
    safety_level: str
    primary_family: str
    def to_dict(self) -> dict[str, object]: ...

def load_taxonomy(labels_path: Path, families_path: Path) -> Taxonomy: ...
```

这些接口代码块是要实现的函数签名，函数体在各任务给定算法与测试约束下编写。types.py 只保存无副作用的数据类型，序列化为标准 JSON 类型，禁止存客户端或密钥。

- [ ] 新建配置行为测试，读取默认配置，并用临时文件覆盖空对象、重复键、空值、未知 family 标签与缺失映射；每种非法情况断言 ConfigError。

```python
def test_taxonomy_contains_distinct_neutral_and_no_emotion():
    taxonomy = load_taxonomy(Path("config/emotion_labels.json"),
                             Path("config/emotion_families.json"))
    assert len(taxonomy.labels) == 34
    assert taxonomy.families["neutral"] == "neutral"
    assert taxonomy.families["no_emotion"] == "no_emotion"
    assert set(taxonomy.labels) == set(taxonomy.families)
    assert taxonomy.labels["neutral"] != taxonomy.labels["no_emotion"]

def test_duplicate_json_keys_are_rejected(tmp_path):
    labels = tmp_path / "labels.json"
    families = tmp_path / "families.json"
    labels.write_text('{"sad":"悲伤","sad":"失落"}', encoding="utf-8")
    families.write_text('{"sad":"sadness_loss"}', encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_taxonomy(labels, families)
```

- [ ] 运行 `python -m pytest tests/emotion/test_config.py -q`，确认因新模块/功能不存在而失败。
- [ ] 从归档 `chatbot/emotion/labels.py` 逐项迁入原有标签描述和 family，不在运行时 import 归档。增加设计中的 neutral/no_emotion 定义。用 object_pairs_hook 拒绝重复键；检查 labels/families 键集合相同。配置必须非空，但不能把数量 34 写成加载器限制，以允许未来增改。

```python
def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate key: {key}")
        result[key] = value
    return result

# 校验后对实际配置计算快照哈希；字典键顺序不影响哈希。
payload = json.dumps({"labels": labels, "families": families},
                     sort_keys=True, ensure_ascii=False, separators=(",", ":"))
content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] 运行配置测试，确认同内容不同排版哈希相同、修改描述哈希改变。
- [ ] `git diff --check` 后仅提交上述文件：`feat: configure emotion taxonomy and families`。

## Task 2: 增量迁移和分析仓储

**Files:** Modify `chatbot/db/schema.py`, `tests/db/test_schema.py`; Create `chatbot/db/emotions.py`, `tests/db/test_emotions.py`; extend `chatbot/emotion/types.py`.

**Interfaces:** `EmotionAnalysis` 为 frozen dataclass：id、conversation_id、request_id、user_message_id、status、created_at、started_at、completed_at 及下述所有 JSON/遥测列；JSON 字段读出为字典/列表，可空字段保留 None。仓储接口：

```python
class EmotionRepository:
    def __init__(self, database: Database): ...
    async def reserve(self, conversation_id: str, request_id: str,
                      user_message_id: str) -> tuple[EmotionAnalysis, bool]: ...
    async def get(self, analysis_id: str) -> EmotionAnalysis | None: ...
    async def find_for_message(self, user_message_id: str) -> EmotionAnalysis | None: ...
    async def start(self, analysis_id: str, *, snapshot: dict[str, object]) -> None: ...
    async def finish(self, analysis_id: str, *, status: str,
                     facts: dict[str, object]) -> None: ...
    async def fail_interrupted(self, *, cutoff: str) -> int: ...
    async def recent_labels(self, conversation_id: str, *,
                            before_sequence: int, limit: int = 3) -> list[str]: ...
```

- [ ] 新增真实 SQLite 用例：通过 IdentityService.resolve 创建会话，MessageRepository.reserve_turn 创建用户消息，测试 reserve 幂等和完成事实留存。

```python
async def test_reserve_is_idempotent_and_bound_to_user(database):
    conversation = (await IdentityService(database).resolve("audit")).conversation
    turn = await MessageRepository(database).reserve_turn(conversation.id, "r1", "你好")
    repo = EmotionRepository(database)
    first, created = await repo.reserve(conversation.id, "r1", turn.user.id)
    second, repeated = await repo.reserve(conversation.id, "r1", turn.user.id)
    assert created is True and repeated is False
    assert first.id == second.id
    with pytest.raises(InvalidMessageState):
        await repo.reserve(conversation.id, "r1", turn.assistant.id)
```

- [ ] 构造 version 1 数据库（执行保留的 V1 DDL，再写入用户/会话/消息），断言升级后原行逐字段不变、版本为 2、重复初始化无副作用；未来版本测试由硬编码 2 改为 SCHEMA_VERSION + 1。运行 `python -m pytest tests/db/test_schema.py tests/db/test_emotions.py -q` 确认失败。
- [ ] 保留旧 DDL 为 V1_DDL_STATEMENTS，新建 V2_DDL_STATEMENTS；同一 immediate transaction 内依次应用迁移，最后设置 user_version=2。保留原有“先只读检查未来版本”的行为。

```sql
CREATE TABLE emotion_analyses (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  request_id TEXT NOT NULL,
  user_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
  snapshot_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT,
  raw_output TEXT,
  reasoning_content TEXT,
  response_metadata_json TEXT,
  error_json TEXT,
  input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
  output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
  total_tokens INTEGER CHECK(total_tokens IS NULL OR total_tokens >= 0),
  latency_ms INTEGER CHECK(latency_ms IS NULL OR latency_ms >= 0),
  finish_reason TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  UNIQUE(conversation_id, request_id)
);
CREATE INDEX ix_emotion_status_created ON emotion_analyses(status, created_at);
```

- [ ] 实现 reserve 在事务中校验消息属于指定会话/请求、role=user 且 completed，返回 `(record, created)`。start 仅允许 pending→running，保存 snapshot 并提交；finish 允许 pending/running→终态，不覆盖已有终态。JSON 编码统一为标准 JSON，未知事实用 null。完整性约束/数据库异常保留既有领域错误边界。
- [ ] 测试重复 finish 不覆盖、错误身份拒绝、snapshot 在新连接立即可读、NULL 用量不变成 0、迁移回滚及 recent_labels 不读未来结果。运行两组测试并提交：`feat: persist emotion analysis audit records`。

## Task 3: 角色化历史、token 预算与整轮裁剪

**Files:** Modify `chatbot/db/messages.py`; Create `chatbot/emotion/history.py`, `budget.py`, `tests/emotion/test_history.py`, `test_budget.py`; extend `types.py`.

**Interfaces:**

```python
# MessageRepository 新接口，返回完整 Message 行而非现有 list_context 的 role/content。
async def list_emotion_history(self, conversation_id: str, *,
                               through_sequence: int) -> list[Message]: ...

@dataclass(frozen=True)
class DialogueTurn:
    request_id: str
    messages: tuple[BaseMessage, BaseMessage]
    message_ids: tuple[str, str]

@dataclass(frozen=True)
class HistorySelection:
    messages: list[BaseMessage]
    audit: dict[str, object]

class ContextBudgetExceeded(ValueError):
    def __init__(self, audit: dict[str, object]):
        super().__init__("context_budget_exceeded")
        self.audit = audit

def group_history(rows: Sequence[Message], *, current_id: str
                  ) -> tuple[list[DialogueTurn], HumanMessage, list[str]]: ...
def select_history(turns: Sequence[DialogueTurn], current: HumanMessage, *,
                   counter: TokenCounter, budget: BudgetConfig,
                   system_messages: Sequence[BaseMessage]) -> HistorySelection: ...
```

- [ ] 写确定性计数器测试，不依赖真实模型 token 数；计数器必须计算消息开销，记录 `.identity` 和 `.version`。

```python
class UnitCounter:
    identity = "unit-test-message-counter"
    version = "1"
    def count(self, messages):
        return sum(len(message.content) + 1 for message in messages)

def test_budget_keeps_recent_pairs_and_current():
    turns = [DialogueTurn(str(i), (HumanMessage(content="aaa", id=f"u{i}"),
                                  AIMessage(content="bbb", id=f"a{i}")),
                          (f"u{i}", f"a{i}")) for i in range(3)]
    result = select_history(turns, HumanMessage(content="now", id="current"),
        counter=UnitCounter(), budget=BudgetConfig(40, 8, 2), system_messages=[])
    assert [m.id for m in result.messages] == ["u1", "a1", "u2", "a2", "current"]
    assert [m.type for m in result.messages] == ["human", "ai", "human", "ai", "human"]
    assert result.audit["chat_tokens"] == 20
    assert result.audit["history_token_cap"] == 24
```

- [ ] 增加整轮恰好 60%、再加一轮超限、最近长轮不跳选更早短轮、当前输入单独超限、system 占用压低预算、40 条以上仍可全部保留用例。运行 `python -m pytest tests/emotion/test_history.py tests/emotion/test_budget.py -q` 确认失败。
- [ ] 数据查询使用 `conversation_id=? AND sequence_no<=? ORDER BY sequence_no`，不使用 LIMIT 条数。配对按 request_id，历史用户和助手均 completed，user 序号早于 assistant；当前用户单独保留。历史不完整时整个 request 排除，返回被排除消息 ID。
- [ ] 实现整轮选择，逐次计算完整候选而非把各条 token 直接相加；固定消息模板在完整请求中的开销可能不同。保留末尾 current HumanMessage，遇超限 break。

```python
cap = int(budget.context_tokens * budget.history_ratio)
selected = [current]

def fits(messages):
    return (counter.count(messages) <= cap and
            counter.count([*system_messages, *messages]) + budget.output_tokens
            + budget.safety_tokens <= budget.context_tokens)

if not fits(selected):
    raise ContextBudgetExceeded({"code": "context_budget_exceeded", "history_token_cap": cap})
for turn in reversed(turns):
    candidate = [*turn.messages, *selected]
    if not fits(candidate):
        break
    selected = candidate
```

- [ ] audit 保存 C、ratio、P、O、M、有效预算 B、计数器标识和版本、保留/移除消息 ID、chat_tokens、prompt_tokens。system 非空时用最终完整请求计数作权威复核；超限异常也保存已知预算。角色序列化输出 `human/assistant` 审计及 `user/assistant` API 快照，不能把消息拼成无角色字符串。
- [ ] 运行历史、预算及 `tests/db/test_messages.py`，通过后提交：`feat: select complete dialogue turns within emotion token budget`。

## Task 4: few-shot 提示词与严格结果解析

**Files:** Create `chatbot/emotion/retrieval.py`, `prompt.py`, `parsing.py`, `config/emotion_examples.json`, `tests/emotion/test_prompt.py`, `test_parsing.py`; extend `types.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class PreparedAnalysis:
    messages: list[BaseMessage]
    snapshot: dict[str, object]

def load_examples(path: Path, taxonomy: Taxonomy) -> list[dict[str, str]]: ...
def select_examples(examples: Sequence[dict[str, str]], dialogue: str,
                    likely_emotions: Sequence[str], limit: int = 4
                    ) -> list[dict[str, object]]: ...
def prepare_analysis(rows: Sequence[Message], *, current_id: str,
                     taxonomy: Taxonomy, examples: Sequence[dict[str, str]],
                     recent_labels: Sequence[str], counter: TokenCounter,
                     budget: BudgetConfig) -> PreparedAnalysis: ...
def parse_result(raw: str, taxonomy: Taxonomy) -> EmotionResult: ...
```

- [ ] 测试 no_emotion 成功和冲突校验，neutral 描述不同，非法主次标签、NaN/bool 置信度、缺失字段、非法 safety_level 全部拒绝。

```python
def test_no_emotion_is_a_successful_classification():
    taxonomy = load_taxonomy(Path("config/emotion_labels.json"), Path("config/emotion_families.json"))
    payload = {"primary_emotion": "no_emotion", "confidence": 0.9,
        "secondary_emotions": [], "evidence": "转换文件", "reply_strategy": "直接完成任务",
        "trajectory_note": "", "safety_level": "normal"}
    assert parse_result(json.dumps(payload), taxonomy).primary_family == "no_emotion"
    payload["secondary_emotions"] = ["sad"]
    with pytest.raises(ValueError):
        parse_result(json.dumps(payload), taxonomy)
```

- [ ] 运行 `python -m pytest tests/emotion/test_prompt.py tests/emotion/test_parsing.py -q` 确认失败。
- [ ] 将归档 v1 examples 的实际示例迁入 JSON，每项 `id/dialogue/emotion`；补 neutral/no_emotion 示例。保持加权词项重合度算法、先验 +2.0、最多 4 条和标签多样性规则；排序相同时用原始索引稳定排序。示例标签不存在、空内容或重复 ID 报 ConfigError。
- [ ] prepare_analysis 先用仅聊天 60% 上限选择候选历史，再基于该候选和最近 3 个不同成功标签检索示例；固定示例后构建 SystemMessage，重新在完整预算内选最近轮次。记录候选和最终 ID。此前被排除的历史不能重新加入。提示词文本固定版本 `emotion-v2-1`，写明目标为最后一条 human，助手文本仅为背景，no_emotion 不等于失败。

```python
instructions = """判断最后一条 human 消息表达的情绪。历史 assistant 消息仅提供背景。
从下方配置选取标签。neutral 表示明确的中性状态；no_emotion 表示未表达情绪。
仅返回 JSON，包含 primary_emotion、confidence、secondary_emotions、evidence、
reply_strategy、trajectory_note、safety_level。confidence 必须在 0 到 1 之间。
no_emotion 不与其他情绪同时出现。safety_level 仅为 normal、supportive 或 crisis。
"""
# 组装：SystemMessage(instructions + 标签说明 + 已选示例) + 选中角色化聊天消息。
```

- [ ] parse_result 接受单个 JSON 对象或完整 JSON 代码围栏，拒绝任意文本中贪婪提取对象和 Emotion: 旧格式。所有约定字段必须存在且类型正确；secondary 不重复且不含 primary。primary_family 由配置查表生成，不采信模型自报 family。原始字符串由调用模块保留，不由解析器修改。
- [ ] 验证 snapshot 含标签/family/hash、示例库快照/hash、所选示例分数及理由、先验、检索参数/版本、提示词/模板版本、历史预算记录。运行情绪纯函数测试并提交：`feat: build auditable emotion prompts and validate results`。

## Task 5: 独立模型配置、token 计数器与诊断边界

**Files:** Create `chatbot/emotion/model.py`, `tests/emotion/test_model.py`; Modify `chatbot/emotion/config.py`, `types.py`, `.env.example`; Test `tests/emotion/test_config.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class EmotionSettings:
    api_key: SecretStr
    model: str
    base_url: str | None
    temperature: float
    timeout_seconds: float
    budget: BudgetConfig
    tokenizer_model: str
    labels_path: Path
    families_path: Path
    examples_path: Path

@dataclass(frozen=True)
class ModelOutcome:
    raw_output: str | None
    reasoning_content: str | None
    usage: TokenUsage
    finish_reason: str | None
    metadata: dict[str, object]
    error: dict[str, object] | None

class EmotionModel(Protocol):
    parameters: dict[str, object]
    async def invoke(self, messages: Sequence[BaseMessage]) -> ModelOutcome: ...

def load_emotion_settings(chat_config: AppConfig) -> EmotionSettings: ...
class OpenAICompatibleEmotionModel:  # 实现 EmotionModel
    def __init__(self, settings: EmotionSettings, *, client=None): ...
class ModelTokenCounter:  # 实现 TokenCounter
    def __init__(self, client: ChatOpenAI, *, tokenizer_model: str): ...
```

- [ ] 注入 FakeClient（`async ainvoke(messages)` 返回 AIMessage 或抛异常）测试调用参数、成功响应和 error 对象。测试 provider 异常带 status_code/request_id/body，验证诊断保留且 API key、认证 URL 和 Authorization 被脱敏。CancelledError 必须向上传播。

```python
class ProviderFailure(Exception):
    status_code = 429
    request_id = "provider-r1"
    body = {"error": {"code": "rate_limit", "message": "quota exceeded"}}

class FailingClient:
    async def ainvoke(self, messages):
        raise ProviderFailure("quota exceeded")

# settings 由该测试文件的 fixture 构造，budget=(4096,512,64)，临时 SecretStr，
# model/tokenizer_model 使用安装版计数器已支持的模型标识；不发真实请求。
async def test_provider_failure_preserves_diagnostics(settings):
    outcome = await OpenAICompatibleEmotionModel(settings, client=FailingClient()).invoke(
        [HumanMessage(content="测试")])
    assert outcome.error["http_status"] == 429
    assert outcome.error["provider_request_id"] == "provider-r1"
    assert outcome.raw_output is None
    assert outcome.usage.input_tokens is None
```

- [ ] 运行 `python -m pytest tests/emotion/test_model.py tests/emotion/test_config.py -q` 确认失败。
- [ ] load_emotion_settings 在应用装配时读取，不改变现有 AppConfig 构造签名。新增 EMOTION_LLM_API_KEY/MODEL/BASE_URL，缺省继承聊天连接；TEMPERATURE 默认 0，TIMEOUT_SECONDS 默认继承聊天超时。显式要求 EMOTION_CONTEXT_TOKENS 和 EMOTION_TOKENIZER_MODEL；OUTPUT_TOKENS 默认 1024，SAFETY_TOKENS 默认 256。ratio 固定 0.60；验证 C/O/M 正整数且 O+M<C、温度范围和路径完整性。
- [ ] `.env.example` 标明 C 必须与部署模型一致，不能给未知模型猜窗口。tokenizer_model 必须映射到服务实际聊天模板，禁止未知模型自动回退到通用编码。ModelTokenCounter 启动时计数一组 system/human/assistant 文本作可用性检查；不支持时 ConfigError，README 明确需要适配服务计数方案。
- [ ] 用 ChatOpenAI(streaming=False, max_retries=0, max_tokens=O, temperature=..., timeout=..., tiktoken_model_name=...) 建立独立客户端；实际传参全部放入脱敏 parameters。计数器调用 `get_num_tokens_from_messages`，记录库版本及 tokenizer_model，使用 M 覆盖计数估计与协议开销差异，不能把本地计数宣传为服务端精确值。
- [ ] 捕获模型异常时立刻提取 stage=model、异常类型、脱敏消息、可读状态码/request ID/body 和嵌套响应中的可用用量。用已有 redact_secrets 并补充对已知 api_key 的字面脱敏；不持久化完整请求 headers、客户端 repr 或带局部变量的 traceback。成功保存 AIMessage 原始内容、实际 reasoning_content、metadata、usage；不合成 reasoning。
- [ ] 运行模型与现有 `tests/llm`，确保聊天适配器行为未改变，提交：`feat: add diagnostic emotion model boundary and token counting`。

## Task 6: 情绪子图与先写入后调用

**Files:** Create `chatbot/emotion/graph.py`, `tests/emotion/test_graph.py`; extend `types.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class EmotionRuntime:
    repository: EmotionRepository
    messages: MessageRepository
    model: EmotionModel
    counter: TokenCounter
    budget: BudgetConfig
    taxonomy: Taxonomy
    examples: list[dict[str, str]]

class AnalysisState(TypedDict, total=False):
    conversation_id: str
    request_id: str
    user_message_id: str
    analysis_id: str
    analysis_status: str
    prepared: PreparedAnalysis
    outcome: ModelOutcome
    result: EmotionResult
    error: dict[str, object]

def build_emotion_graph(runtime: EmotionRuntime) -> CompiledStateGraph: ...
```

子图内部 prepared/outcome 仅在不带 checkpointer 的单次调用内使用；父图只接收 ID/status，不能把整份子图返回值合并进父状态。若未来给子图启用 checkpoint，先将内容改为引用，不直接持久化这些运行对象。

- [ ] 写 SpyModel，在 invoke 入口用另一个 SQLite 连接读取 analysis 状态及 snapshot，断言 running 且完整输入参数已提交；输出合法 JSON 后验证 completed、结果与 raw 同时存在。

```python
class AuditSpyModel:
    parameters = {"model": "fake", "max_retries": 0}
    def __init__(self, repo, user_message_id, raw):
        self.repo, self.user_message_id, self.raw = repo, user_message_id, raw
        self.calls = 0
    async def invoke(self, messages):
        row = await self.repo.find_for_message(self.user_message_id)
        assert row.status == "running"
        assert row.snapshot_json["model_parameters"]["max_retries"] == 0
        assert row.snapshot_json["prompt"]
        self.calls += 1
        return ModelOutcome(self.raw, None, TokenUsage(), "stop", {}, None)
```

- [ ] 分别测试解析失败、模型超时、预算超限、开始写入失败、结束写入失败、重复已完成/已失败调用、取消中断。运行 `python -m pytest tests/emotion/test_graph.py -q` 确认失败。
- [ ] prepare 节点先 reserve；读取当前消息和历史边界、近期标签，再 prepare_analysis。已存在终态直接返回，不调用模型。新记录生成 snapshot 后 start 并提交；准备失败以 pending→failed 保存阶段错误和已获取的配置/预算信息。
- [ ] call 节点仅运行 prepared 成功的记录。保存 ModelOutcome；解析错误把原始内容和 validation 错误放入状态。计时覆盖实际分析阶段，模型可获取时间另存 metadata，未知用量保持 NULL。
- [ ] persist 节点构建仓储 facts 并 finish，失败状态也完整保存 snapshot/raw/error。所有程序异常按 prepare/model/parse/validation 分类；SQLite 错误转 DatabaseError 向父图传播，不能转换成普通模型失败继续。取消时尽力写 process_interrupted，随后重新抛 CancelledError。

```python
builder = StateGraph(AnalysisState)
builder.add_node("prepare_analysis", prepare_node)
builder.add_node("call_emotion_model", call_node)
builder.add_node("persist_analysis", persist_node)
builder.add_edge(START, "prepare_analysis")
builder.add_edge("prepare_analysis", "call_emotion_model")
builder.add_edge("call_emotion_model", "persist_analysis")
builder.add_edge("persist_analysis", END)
# 各节点先检查终态/前序 error。compile() 不传 checkpointer。
```

- [ ] 验证测试通过并检查数据库中无凭据、一次逻辑分析最多一次模型调用；非终态旧记录不盲目重试，按中断失败完成。提交：`feat: execute durable emotion analysis subgraph`。

## Task 7: 主图接入、回复上下文与启动恢复

**Files:** Modify `chatbot/graph/state.py`, `dependencies.py`, `nodes.py`, `builder.py`, `chatbot/llm/prompt.py`, `chatbot/web.py`, `chatbot/services/recovery.py`; Create `tests/graph/test_emotion_turn.py`; Modify `tests/graph/test_turn_graph.py`, `tests/services/test_recovery.py`, `tests/api/conftest.py`, `tests/app/test_web.py`, `tests/app/test_acceptance.py`, `tests/app/test_live_llm.py` 及其 create_app 调用点。

**Interfaces:** NodeDependencies 新增必需的 `emotion: EmotionRuntime`；TurnState 新增 `emotion_analysis_id: str | None`、`emotion_status: str | None`，不存整份结果。`TurnNodes.analyze_emotion(state, runtime) -> TurnState`；`build_prompt(messages, *, emotion_context: str | None = None)`；`create_app(config=None, model=None, *, emotion_model=None, emotion_counter=None, emotion_settings=None)` 为测试提供明确注入点，生产不绕过配置检查。

- [ ] 在新图测试文件使用真实 SQLite saver。将现有 `_reserved_context`、RecordingPublisher、DeterministicModel 的最小公共构造移动到 `tests/graph/helpers.py`（同步更新原文件 import，行为不变）。新增 runtime fixture 从临时配置/UnitCounter/假 EmotionModel 构造 EmotionRuntime，更新所有 NodeDependencies 调用，防止旧测试意外连真实服务。
- [ ] 写集成断言：chat 假模型在 stream 开始前查询 emotion 终态；成功 prompt 含本轮 primary，失败 prompt 不含情绪上下文，chat 最终 completed。重复同请求 emotion/chat 调用计数都不增加。

```python
# 在 test_emotion_turn.py，使用本文件创建的 runtime fixture 和公共 _reserved_context。
async def test_failed_analysis_still_completes_chat(database, messages, saver, emotion_runtime):
    context, state = await _reserved_context(database, messages, "fallback")
    class FailedModel:
        parameters = {"model": "fake", "max_retries": 0}
        async def invoke(self, prompt):
            return ModelOutcome(None, None, TokenUsage(), None, {},
                {"stage": "model", "type": "TimeoutError", "message": "timeout"})
    runtime = dataclasses.replace(emotion_runtime, model=FailedModel())
    chat = DeterministicModel([ModelDelta(content="普通回复")])
    graph = compile_turn_graph(NodeDependencies(messages=messages, model=chat, emotion=runtime), saver)
    result = await graph.ainvoke(state, {"configurable": {"thread_id": context.thread_id}}, context=context)
    row = await runtime.repository.find_for_message(context.user_message_id)
    assert row.status == "failed"
    assert row.error_json["type"] == "TimeoutError"
    assert result["phase"] == "completed"
```

- [ ] 运行 `python -m pytest tests/graph tests/services/test_recovery.py -q` 确认新期望失败。
- [ ] 主图边改为 prepare_turn→analyze_emotion→generate_response→finalize_turn。新节点先验证 context/消息身份和 terminal，调用子图后只返回 analysis_id/status。将节点时长、状态、analysis_id 放助手 trace，不复制分析 raw 或覆盖聊天模型审计参数。
- [ ] generate_response 按 ID 从仓储读取并重新校验关联用户消息。只有 completed 分析添加情绪 context；failed 不读上轮。新增 system 段以“模型推断，仅供回复参考”的含义描述结果，用户证据采用 JSON 数据形式，不能被当作新系统指令。普通聊天 build_prompt 默认参数保持兼容。
- [ ] 新节点发生 DatabaseError 时调用现有 assistant 失败保存逻辑并清理 `_runs`，确保 generate_response 不再调用模型；取消时清理并重抛。测试注入结束写失败，不能留下 streaming 或内存事实泄漏。
- [ ] web lifespan 在接收请求前加载 EmotionSettings/Taxonomy/examples、建立计数器和情绪模型、组装 repository/runtime；测试注入全部使用假实现。shutdown 等待上限覆盖情绪 timeout + 聊天 timeout，避免新增串行调用被旧单次 timeout 错误终止。
- [ ] recover_interrupted_turns 增加 emotion repository 参数，并以同一 cutoff 先将未完成分析写入 process_interrupted（保留 snapshot、raw 和已有诊断），再沿用消息失败和 checkpoint 清理；RecoveryReport 增加 failed_analyses。测试纯 pending、running、completed 不被改动及断点清理。
- [ ] 更新原图 STATE_FIELDS、节点序列断言和离线应用 fixture；不删减原有幂等、隔离、流式、失败清理测试。运行 `python -m pytest tests/graph tests/services tests/api tests/app -q`，通过后提交：`feat: run emotion analysis before each chat response`。

## Task 8: 验收、使用说明与最终检查

**Files:** Create `tests/app/test_emotion_flow.py`; Modify `README.md`, `.env.example`；必要修正仅限前述任务范围。

- [ ] 写应用端到端离线用例，通过 `/api/users/resolve` 和现有消息 API 连续发送三条输入，使用成功→失败→成功的假情绪模型；等待助手终态后用新连接查询 emotion_analyses，验证三条记录、失败详细信息、各自提示词快照、第三轮包含第二轮普通回复且不复用第一轮情绪。
- [ ] 写跨会话和 60% 边界验收，用预算计数器捕获实际请求。断言保留 ID 是完整轮次后缀加当前 human，所有 request 的 user/assistant 配对正确，原 messages 表未被删除或改写。

```python
# 收集实际模型入参后，在应用验收用例直接复核，不只相信 audit 数字。
assert counter.count(sent_chat_messages) <= int(settings.budget.context_tokens * 0.60)
assert (counter.count(sent_full_prompt) + settings.budget.output_tokens
        + settings.budget.safety_tokens <= settings.budget.context_tokens)
assert sent_chat_messages[-1].type == "human"
assert sent_chat_messages[-1].id == current_user_message_id
```

- [ ] 运行 `python -m pytest tests/app/test_emotion_flow.py -q`。红灯时先定位集成断点，修正对应模块，再跑该用例至绿；不能通过降低预算、漏存错误或丢弃角色来修正测试。
- [ ] README 加入两个标签配置示例、neutral/no_emotion/family 含义、完整轮次裁剪、token 计数属于本地估计及 M 余量、模型/计数器配置、分析与 DB 失败区别、配置重启生效和审计 SQL：

```sql
SELECT user_message_id, status,
       json_extract(result_json, '$.primary_emotion') AS emotion,
       json_extract(result_json, '$.primary_family') AS family,
       json_extract(error_json, '$.stage') AS failure_stage,
       json_extract(error_json, '$.message') AS failure_message,
       input_tokens, output_tokens, latency_ms
FROM emotion_analyses ORDER BY created_at, id;
```

- [ ] `.env.example` 列出 Task 5 全部变量及三个配置路径变量 EMOTION_LABELS_PATH、EMOTION_FAMILIES_PATH、EMOTION_EXAMPLES_PATH。路径默认以模块推导的项目根 config/ 为基准，避免依赖原目录未提交的 chatbot/core/paths.py。明确用户必须填写实际模型 C 和匹配的 tokenizer_model，生产缺失时启动报错而不是静默禁用情绪。
- [ ] 使用 verification-before-completion 技能，执行 `python -m pytest -q` 和 `git diff --check`。记录实际通过/跳过/失败数量；本计划编写阶段没有运行这些测试，不能预填通过结果。
- [ ] 自行审查迁移、错误脱敏、幂等和完整历史选择；按用户限制不派子代理。确认只改本任务文件，提交验收与文档：`test: verify emotion audit and document configuration`。
- [ ] 检查 `git status --short`、`git diff main...HEAD --stat`、`git log --oneline main..HEAD`；结果中说明实现、测试和已知限制，列出待合并分支/worktree/目标。保留分支，不自动合并。

## 设计覆盖自检

| 设计要求 | 实施任务 |
|---|---|
| 34 个选项、两份配置、快照/hash、family 派生 | 1、4 |
| 逐条分析、幂等、成功后回复、失败普通回复 | 2、6、7、8 |
| 全会话候选、60% 预算、角色和整轮裁剪、超限失败 | 3、4、5、8 |
| 动态示例/先验/检索审计 | 2、4、6 |
| 原始输出、实际推理、用量、参数与失败细节 | 2、5、6 |
| 先保存参数再调用、数据库失败与取消清理 | 2、6、7 |
| v1 迁移、启动恢复、旧消息不自动回填 | 2、7 |
| 不改变聊天窗口、不混入未提交改动 | 全局约束、7、8 |

按 Task 1→8 顺序执行，每项独立红绿验证后提交。计划交付后默认可在本会话串行执行；用户明确要求子代理时才调整执行方式。
