# 情绪感知本地聊天机器人

这是一个本地优先的单用户 Web 聊天机器人。后端使用 FastAPI 和 LangChain 调用 OpenAI-compatible LLM，前端通过 Server-Sent Events（SSE）流式显示回复；聊天历史、用户画像、情绪分析、反馈和长期记忆默认保存在本地 SQLite 数据库中。

项目同时包含一套可复现的情绪识别实验工具：固定 32 类情绪标签、动态示例检索、5 组消融配置、离线指标计算、双语 benchmark，以及可选的 Codex CLI 隔离运行器。

## 主要能力

- 使用无需构建的 HTML、CSS 和 JavaScript 前端进行流式聊天。
- 支持 `openai`、`deepseek` 两类 OpenAI-compatible provider。
- 可为聊天和情绪识别分别配置 LLM；未单独配置时复用同一实例。
- 每隔固定用户回合执行一次结构化情绪识别，并在页面展示当前状态和近期轨迹。
- 将当前情绪、回复策略、安全提示、用户画像和相关长期记忆注入聊天 Prompt。
- 提供可跳过的首次画像录入，以及后续画像查看和编辑。
- 支持 AI 回复点赞、点踩、按原因重新生成，以及情绪识别正确性反馈。
- 使用本地 SQLite 保存长期记忆，执行词法检索、去重、冲突处理和周期性规则提炼。
- 提供 JSON/JSONL 离线评估、OpenAI-compatible LLM 消融和 Codex CLI 消融流程。

## 技术栈与运行要求

- Python 3.10+
- FastAPI、Uvicorn
- LangChain、`langchain-openai`
- Python 标准库 `sqlite3`
- OpenAI-compatible API 凭据
- 可选：已安装并登录的 Codex CLI，仅用于 `run_codex_cli_emotion_ablation.py`

项目没有前端构建步骤，也不依赖 Mem0 Platform、云端向量数据库或第三方托管记忆服务。

## 快速开始

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

复制配置模板：

```bash
cp .env.example .env
```

至少填写聊天 LLM 的 API Key。以下是 DeepSeek 示例：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.7

EMOTION_INTERVAL=5
MEMORY_ENABLED=true
MEMORY_DB_PATH=data/records/memory.sqlite3
MEMORY_MAX_RESULTS=5
PROMPT_CONFIG_PATH=data/config/prompts.json
```

启动应用：

```bash
uvicorn chatbot.web:app
```

打开 <http://127.0.0.1:8000>。

`chatbot.main` 只用于配置校验和 Web 启动提示，不再提供交互式 CLI 聊天。

## 配置

`chatbot.config.load_config()` 按“CLI 参数 > 环境变量 > 默认值”解析 LLM 配置。Web 应用使用 `load_config([])`，因此 Web 运行时通常是“环境变量 > 默认值”。

### 聊天 LLM

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | 当前支持 `openai` 和 `deepseek`。 |
| `LLM_API_KEY` | 无 | 必填；未设置时兼容 `OPENAI_API_KEY`。 |
| `LLM_MODEL` | `gpt-4o-mini` | 未设置时兼容 `OPENAI_MODEL`。 |
| `LLM_BASE_URL` | 空 | OpenAI-compatible API 地址；兼容 `OPENAI_BASE_URL`。 |
| `LLM_TEMPERATURE` | `0.7` | 采样温度；兼容 `OPENAI_TEMPERATURE`。 |

### 情绪识别 LLM

如果所有 `EMOTION_LLM_*` 都未设置，情绪识别会复用聊天 LLM。只要出现任一 `EMOTION_LLM_*` 配置，程序就会创建独立情绪 LLM，空缺字段继承聊天 LLM。

| 变量 | 未设置时 |
| --- | --- |
| `EMOTION_LLM_PROVIDER` | 继承 `LLM_PROVIDER`。 |
| `EMOTION_LLM_API_KEY` | 继承 `LLM_API_KEY`。 |
| `EMOTION_LLM_MODEL` | 继承 `LLM_MODEL`。 |
| `EMOTION_LLM_BASE_URL` | 继承 `LLM_BASE_URL`。 |
| `EMOTION_LLM_TEMPERATURE` | 继承 `LLM_TEMPERATURE`。 |
| `EMOTION_INTERVAL` | `5`，必须为正整数。 |

独立情绪 LLM 示例：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_chat_api_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1

EMOTION_LLM_PROVIDER=openai
EMOTION_LLM_API_KEY=your_emotion_api_key
EMOTION_LLM_MODEL=gpt-4o-mini
EMOTION_LLM_BASE_URL=https://api.openai.com/v1
EMOTION_LLM_TEMPERATURE=0
EMOTION_INTERVAL=5
```

### 本地长期记忆

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MEMORY_ENABLED` | `true` | `0`、`false`、`no`、`off` 会关闭记忆。 |
| `MEMORY_DB_PATH` | `data/records/memory.sqlite3` | 长期记忆数据库路径。 |
| `MEMORY_MAX_RESULTS` | `5` | 每轮最多注入的相关记忆数；非法值回退到默认值。 |
| `MEMORY_CONSOLIDATION_ENABLED` | `true` | 是否启用周期性记忆提炼；长期记忆关闭时也会关闭。 |
| `MEMORY_CONSOLIDATION_INTERVAL` | `5` | 两次提炼之间的用户回合数。 |
| `MEMORY_CONSOLIDATION_WINDOW` | `12` | 每次最多检查的近期 human/AI 消息数。 |
| `MEMORY_CONSOLIDATION_MODE` | `rules` | 当前仅支持本地规则模式，非法值回退为 `rules`。 |

单轮抽取会保守识别用户明确表达的偏好、画像、目标和边界；周期性提炼用于识别重复压力源和跨轮稳定偏好。SQLite provider 会合并重复记忆、保留更强边界，并只检索仍处于 active 状态的记录。

### Prompt 模板

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PROMPT_CONFIG_PATH` | `data/config/prompts.json` | 自定义聊天和情绪分析 Prompt。 |

可以复制 `data/config/prompts.example.json`：

```bash
cp data/config/prompts.example.json data/config/prompts.json
```

支持两个字段：

- `chat_system`：聊天 system Prompt 主体；画像、记忆和情绪上下文仍由程序追加。
- `emotion_analysis`：情绪识别模板，可使用 `{emotion_labels}`、`{example_block}`、`{likely_line}`、`{dialogue_context}`。

文件缺失、JSON 无效、字段为空或模板占位符错误时，程序会回退到内置默认 Prompt。

## 一次聊天请求如何运行

1. 前端 `POST /api/chat/streams` 提交消息并取得一次性 `stream_id`。
2. 前端使用 `EventSource` 消费 `/api/chat/streams/{stream_id}`。
3. `ChatService` 将用户消息写入 `runtime.sqlite3`。
4. 服务使用当前输入、当前情绪和最近情绪检索本地长期记忆。
5. 到达 `EMOTION_INTERVAL` 时，情绪 LLM 分析最近对话并保存结构化结果。
6. `safety.py` 根据当前输入和情绪状态补充 `normal`、`supportive` 或 `crisis` 级别的回复提示。
7. 用户画像、相关长期记忆和情绪上下文被注入聊天 Prompt，聊天 LLM 开始流式生成。
8. 完整 AI 回复写入历史，并生成可用于反馈和重新生成的消息 ID。
9. 服务从本轮内容保守抽取长期记忆；到达间隔时再执行一次跨轮提炼。

记忆检索、记忆写入、记忆提炼或情绪识别失败不会阻断正常聊天；错误会通过 warning、SSE 状态或接口错误暴露。

## 情绪识别

标签集合定义在 `chatbot/emotion_labels.py`，共 32 类。默认情绪 Prompt 使用：

- 最近最多 `EMOTION_INTERVAL` 轮对话；
- 基于词法重叠和近期情绪先验选择的动态 EICL 示例；
- 最近 3 个成功情绪标签形成的候选先验。

LLM 应返回结构化 JSON：

```json
{
  "primary_emotion": "anxious",
  "confidence": 0.82,
  "secondary_emotions": ["hopeful"],
  "evidence": "I am worried about tomorrow's demo but I still want to try.",
  "reply_strategy": "Acknowledge the worry and suggest one small next step.",
  "trajectory_note": "Anxiety remains, with some hopeful intent.",
  "safety_level": "normal"
}
```

解析器会校验主情绪和次级情绪、把置信度限制在 0 到 1，并将未知安全级别回退为 `normal`。为兼容旧结果，`Emotion: anxious` 等行式输出仍可解析为最小状态。

安全层是本地关键词和高置信度情绪驱动的回复策略，不是临床评估、诊断或紧急服务。

## HTTP API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/` | 返回聊天页面。 |
| `GET` | `/api/history?limit=10` | 返回最近 human/AI 消息。 |
| `GET` | `/api/session?limit=10` | 返回最近消息和与当前历史匹配的最新情绪。 |
| `GET` | `/api/profile` | 返回用户画像及是否为空。 |
| `PUT` | `/api/profile` | 保存经过字段过滤的用户画像并刷新聊天链。 |
| `GET` | `/api/profile/onboarding/questions` | 返回 5 个可跳过的画像问题。 |
| `POST` | `/api/profile/onboarding/draft` | 用聊天 LLM 生成画像草稿；不可用时回退为规则草稿。 |
| `GET` | `/api/emotion/timeline?limit=10` | 返回与当前历史匹配的近期结构化情绪状态。 |
| `POST` | `/api/chat/streams` | 创建一次性聊天流 ID。 |
| `GET` | `/api/chat/streams/{stream_id}` | 消费 SSE 流；同一 ID 只能使用一次。 |
| `POST` | `/api/messages/{message_id}/feedback` | 保存 `like` 或 `dislike`。 |
| `POST` | `/api/messages/{message_id}/regenerate` | 按固定原因重新生成一条 AI 回复。 |
| `POST` | `/api/emotion/feedback` | 保存情绪识别正确性反馈。 |

SSE 事件：

| 事件 | 含义 |
| --- | --- |
| `user_message` | 用户消息已写入。 |
| `emotion_start` | 开始情绪分析。 |
| `emotion_done` | 情绪分析成功，包含 state 和 safety。 |
| `emotion_error` | 情绪分析失败，本轮聊天继续。 |
| `token` | 一段回复内容。 |
| `done` | 回复完成，包含完整内容和消息元数据。 |
| `error` | 输入无效或聊天生成失败。 |

## 本地数据

默认运行时文件均位于不会提交到 Git 的 `data/records/`：

| 文件 | 内容 |
| --- | --- |
| `data/records/runtime.sqlite3` | `chat_history`、`emotion_analysis`、`emotion_feedback` 等 JSON namespace，以及 `profile_entries` 用户画像表。 |
| `data/records/memory.sqlite3` | 长期记忆、状态、替代关系、使用次数和记忆提炼 checkpoint。 |

聊天历史、情绪状态和用户画像不会再从旧 JSON 文件加载，也没有自动迁移逻辑。离线评估脚本也不会直接读取 SQLite；需要向它提供 JSON 或 JSONL 实验结果。

## 离线评估与消融

### 1. 评估一份 JSON/JSONL 结果

```bash
python scripts/evaluate_emotion_analysis.py \
  --analysis-file data/examples/dynamic_eicl_sample.json \
  --labels-file data/examples/emotion_labels_sample.json
```

`--analysis-file` 和 `--labels-file` 都支持 JSON 数组或 JSONL。标注字段可使用 `expected`、`emotion` 或 `label`；记录依次按 `id/case_id`、`index`、`turn_count`、`timestamp` 或位置匹配。

输出包含样本数、正确数、Accuracy、Macro F1 和错误样例。

仓库保留了三份小型示例：

- `data/examples/emotion_labels_sample.json`
- `data/examples/static_few_shot_sample.json`
- `data/examples/dynamic_eicl_sample.json`

### 2. 运行 5 组 OpenAI-compatible LLM 消融

`scripts/run_emotion_ablation.py` 复用应用的情绪 LLM 配置并运行：

| Run | 示例策略 | 情绪历史先验 | 上下文 |
| --- | --- | --- | --- |
| `full` | 动态示例 | 开启 | 默认窗口 |
| `no_dynamic_examples` | 静态示例 | 开启 | 默认窗口 |
| `no_emotion_history` | 动态示例 | 关闭 | 默认窗口 |
| `short_context` | 动态示例 | 开启 | 最近 1 轮 |
| `zero_shot` | 无示例 | 关闭 | 默认窗口 |

生成结果：

```bash
python scripts/run_emotion_ablation.py \
  --dialogues-file data/examples/ablation_dialogues.jsonl \
  --output-dir data/records/ablation
```

汇总指标：

```bash
python scripts/evaluate_emotion_ablation.py \
  --labels-file data/examples/ablation_labels.jsonl \
  --run full=data/records/ablation/full.json \
  --run no_dynamic_examples=data/records/ablation/no_dynamic_examples.json \
  --run no_emotion_history=data/records/ablation/no_emotion_history.json \
  --run short_context=data/records/ablation/short_context.json \
  --run zero_shot=data/records/ablation/zero_shot.json \
  --markdown-file data/records/ablation/summary.md \
  --csv-file data/records/ablation/metrics.csv
```

`zero_shot` 同时去掉 few-shot 示例和情绪历史先验，因此是组合消融，不能把指标变化单独归因于其中一个组件。

### 3. Emotion Ablation V2 双语数据集

`data/benchmarks/emotion_ablation_v2/` 当前版本为 `0.1.0`：

- 500 条确定性生成的正式 release：`core_parallel=256`、`extended_independent=180`、`challenge=64`；
- 64 条 seed 参考集；
- 中文 250 条、英文 250 条，覆盖全部 32 个目标标签；
- validator、双语平行检查、分布汇总、导出和确定性生成脚本。

正式 release 的 `expected` 是生成器目标标签，`label_provenance=synthetic_generator_target`，不是独立人工标注或仲裁后的 ground truth。`annotation/` 中的空文件只是未来双人标注流程的占位符。

常用命令：

```bash
python scripts/benchmark/validate_emotion_benchmark.py \
  --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl

python scripts/benchmark/check_parallel_equivalence.py \
  --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl

python scripts/benchmark/summarize_emotion_benchmark.py \
  --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl \
  --output-dir data/benchmarks/emotion_ablation_v2/reports

python scripts/benchmark/export_emotion_ablation_v2.py \
  --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl \
  --output-dir data/records/ablation_v2_export
```

数据格式和标注语义详见 `data/benchmarks/emotion_ablation_v2/README.md`。

### 4. 可选：使用 Codex CLI 运行隔离消融

先把 seed 集导出为现有消融格式：

```bash
python scripts/benchmark/export_emotion_ablation_v2.py \
  --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl \
  --output-dir data/records/ablation_v2_seed_export
```

执行 10 条 pilot。`--model` 必填，请替换为当前 Codex CLI 支持的模型：

```bash
python scripts/run_codex_cli_emotion_ablation.py \
  --dialogues-file data/records/ablation_v2_seed_export/dialogues.jsonl \
  --output-dir data/records/codex_cli_ablation/pilot \
  --limit 10 \
  --model YOUR_CODEX_MODEL
```

去掉 `--limit 10` 并更换输出目录即可运行全部 64 条 seed：

```bash
python scripts/run_codex_cli_emotion_ablation.py \
  --dialogues-file data/records/ablation_v2_seed_export/dialogues.jsonl \
  --output-dir data/records/codex_cli_ablation/seed64 \
  --model YOUR_CODEX_MODEL
```

运行器为每个“样本 × 配置”启动独立的 `codex exec --ephemeral --sandbox read-only`，默认超时 180 秒、失败最多重试 1 次，并在每个任务后原子保存结果。只有 Prompt、模型、Schema、Codex CLI 版本和运行环境 provenance 全部一致时，已有成功结果才会被复用。

生成中文报告：

```bash
python scripts/report_codex_cli_emotion_ablation.py \
  --seed-file data/benchmarks/emotion_ablation_v2/release/seed.jsonl \
  --run full=data/records/codex_cli_ablation/seed64/full.json \
  --run no_dynamic_examples=data/records/codex_cli_ablation/seed64/no_dynamic_examples.json \
  --run no_emotion_history=data/records/codex_cli_ablation/seed64/no_emotion_history.json \
  --run short_context=data/records/codex_cli_ablation/seed64/short_context.json \
  --run zero_shot=data/records/codex_cli_ablation/seed64/zero_shot.json \
  --output-dir data/records/codex_cli_ablation/seed64
```

报告器会生成 `metrics.csv`、`summary.md` 和 `report-zh.md`，把调用失败计入正式指标，并检查各消融 Prompt 是否真的区别于 `full`。

## 项目结构

```text
chatbot/
  web.py                  FastAPI 应用、HTTP 路由和 SSE
  chat_service.py         聊天、情绪、安全、记忆和持久化编排
  config.py               聊天与情绪 LLM 配置
  llm.py                  LangChain Prompt、chain 和会话历史
  llm_adapter.py          OpenAI-compatible LLM 适配器

  emotion.py              情绪 Prompt 调用和分析记录
  emotion_state.py        结构化情绪状态解析与时间线
  emotion_prompt.py       情绪识别 Prompt 组装
  emotion_retrieval.py    动态 EICL 示例选择
  emotion_feedback.py     情绪正确性反馈
  safety.py               本地安全级别与回复提示

  runtime_store.py        通用 SQLite 运行时存储
  history.py              消息、点赞点踩和重新生成记录
  profile.py              用户画像持久化
  profile_onboarding.py   画像问题、草稿和字段过滤

  memory.py               长期记忆协议和配置
  local_memory.py         SQLite 记忆、检索、去重与冲突处理
  memory_extractor.py     单轮保守记忆抽取
  memory_consolidation.py 周期性跨轮记忆提炼
  prompt_config.py        可覆盖 Prompt 与内置回退
  static/                 无构建前端

scripts/
  evaluate_emotion_analysis.py         单组离线指标
  run_emotion_ablation.py              OpenAI-compatible LLM 消融
  evaluate_emotion_ablation.py         多组消融汇总
  run_codex_cli_emotion_ablation.py    Codex CLI 隔离运行器
  report_codex_cli_emotion_ablation.py Codex 消融报告
  benchmark/                           V2 数据集工具

data/
  config/                 Prompt 示例和 Codex 输出 Schema
  examples/               小型评估与消融样例
  benchmarks/             Emotion Ablation V2
  records/                本地运行时数据与实验输出，不提交到 Git

tests/                    单元、Web、前端脚本和实验工具测试
```

## 测试

安装依赖后运行完整测试：

```bash
python -m pytest -q
```

只检查 README 中的关键路径和样例文件：

```bash
python -m pytest tests/test_readme.py -q
```

## 当前边界

- 应用固定使用 `default` session，面向本地单用户场景，没有认证、多用户隔离或生产级权限控制。
- 一次性 SSE stream id 保存在当前进程内存中；进程重启后失效，同一 id 只能消费一次。
- `runtime.sqlite3` 路径由代码中的默认值确定；只有长期记忆数据库提供 `MEMORY_DB_PATH` 环境变量。
- 关键词安全层只调整回复策略，不能替代专业心理健康服务或紧急援助。
- V2 正式 release 是确定性合成 benchmark，不能表述为人工标注真实分布。
