# 情绪识别聊天机器人

这是一个本地优先的 FastAPI Web 聊天机器人。应用用 OpenAI-compatible LLM 生成回复，通过浏览器端 SSE 流式显示结果，并按固定用户轮数分析情绪。聊天历史、用户画像、情绪分析记录、AI 消息反馈、情绪反馈和长期记忆都默认保存在本地 `data/records/` 的 SQLite 数据库中。

项目当前支持 `openai` 和 `deepseek` 两类 OpenAI-compatible provider。长期记忆使用 Python 标准库 `sqlite3`，不依赖 Mem0 Platform、云端向量库或第三方托管存储。

## 快速开始

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

复制环境变量示例：

```bash
cp .env.example .env
```

编辑 `.env`。DeepSeek 示例：

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

启动 Web 应用：

```bash
uvicorn chatbot.web:app
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 功能概览

- FastAPI 服务和免构建的 HTML/CSS/JavaScript 前端。
- 使用一次性 stream id 建立 SSE 连接，避免浏览器自动重连时重复提交同一条消息。
- 流式生成聊天回复，并在生成完成后写入 AI 消息记录。
- 从本地 SQLite 恢复聊天历史和最近情绪状态。
- 每隔 `EMOTION_INTERVAL` 个用户回合调用情绪 LLM，保存结构化情绪状态。
- 将当前情绪、危机/支持性安全提示和相关长期记忆注入聊天 prompt。
- 首次无用户画像时，可通过“我的画像”入口回答轻量问题，由 LLM 生成可编辑画像草稿，确认后保存。
- 支持 AI 消息点赞、点踩和重新生成。
- 支持情绪识别正确性反馈。
- 使用 SQLite 保存长期记忆，包括偏好、稳定画像、长期目标和明确约束。
- 提供情绪识别评估脚本和 ablation 对比脚本。

## 运行流程

应用启动时，`chatbot.web.build_service()` 会完成这些步骤：

1. 通过 `chatbot.config.load_config()` 读取 `.env`、环境变量和默认值。
2. 从 `data/records/runtime.sqlite3` 恢复聊天历史。
3. 从 `data/records/runtime.sqlite3` 读取可选静态用户画像。
4. 构建聊天 LLM；如果配置了任意 `EMOTION_LLM_*`，再构建独立情绪 LLM。
5. 构建本地 memory provider，默认数据库是 `data/records/memory.sqlite3`。
6. 将历史消息恢复到 LangChain 的 `InMemoryChatMessageHistory`。
7. 构建聊天 chain，并创建单用户 `ChatService`。

如果用户画像为空，前端会显示非阻塞画像录入提示。用户可以跳过继续聊天，也可以通过“我的画像”回答五个轻量问题，确认 LLM 生成的画像草稿后写入 `profile_entries`。

用户发送消息时：

1. 前端 `POST /api/chat/streams` 创建一次性 `stream_id`。
2. 前端用 `EventSource` 连接 `/api/chat/streams/{stream_id}` 消费 SSE。
3. `ChatService.stream_reply()` 写入 human 消息并刷新长期记忆上下文。
4. 到达情绪识别间隔时，调用情绪 LLM，写入情绪分析记录，并更新当前情绪状态。
5. 对当前输入做安全提示判断；危机或支持性提示会覆盖回复策略。
6. 将 `memory_context` 和 `emotion_context` 注入聊天 prompt。
7. 聊天 LLM 流式返回 token，前端逐段渲染。
8. 完整回复写入历史，生成可反馈的 AI 消息 id。
9. 从本轮用户消息和 AI 回复中保守抽取长期记忆候选，写入 SQLite。

记忆检索、记忆写入或情绪识别失败时，聊天会继续；失败会以 warning 或 SSE 状态暴露。

## HTTP 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/` | 返回静态聊天页面。 |
| `GET` | `/api/history?limit=10` | 返回最近 human/AI 消息。 |
| `GET` | `/api/session?limit=10` | 返回最近消息和当前可匹配的最新情绪。 |
| `GET` | `/api/profile` | 返回当前用户画像和是否为空。 |
| `PUT` | `/api/profile` | 保存用户确认后的画像，并刷新后续聊天使用的画像 prompt。 |
| `GET` | `/api/profile/onboarding/questions` | 返回首次画像录入的固定轻量问题。 |
| `POST` | `/api/profile/onboarding/draft` | 根据用户回答生成可编辑画像草稿；LLM 不可用时返回规则草稿。 |
| `GET` | `/api/emotion/timeline?limit=10` | 返回最近情绪状态时间线。 |
| `POST` | `/api/chat/streams` | 创建一次性聊天 stream id。 |
| `GET` | `/api/chat/streams/{stream_id}` | 消费 SSE 聊天流；同一 id 只能消费一次。 |
| `POST` | `/api/messages/{message_id}/feedback` | 对 AI 消息保存 `like` 或 `dislike`。 |
| `POST` | `/api/messages/{message_id}/regenerate` | 按原因重新生成某条 AI 回复。 |
| `POST` | `/api/emotion/feedback` | 保存情绪识别正确性反馈。 |

SSE 事件包括：

- `user_message`：用户消息已写入。
- `emotion_start`：开始情绪分析。
- `emotion_done`：情绪分析成功，返回结构化状态。
- `emotion_error`：情绪分析失败，本轮聊天继续。
- `token`：聊天回复片段。
- `done`：回复完成，包含完整内容和可选消息元数据。
- `error`：聊天生成失败或请求无效。

## 项目结构

```text
chatbot/
  web.py                FastAPI 应用、路由、静态页面和 SSE 接口
  chat_service.py       单用户聊天编排：历史、记忆、情绪分析、安全提示和回复生成
  config.py             LLM 和情绪分析间隔配置
  main.py               运行时 LLM 构建和 CLI 提示入口
  llm.py                LangChain prompt、chain 和会话历史
  llm_adapter.py        OpenAI-compatible LLM 适配层

  emotion.py            情绪 prompt 构造、LLM 调用、解析和记录写入
  emotion_state.py      结构化情绪状态解析、格式化和时间线生成
  emotion_prompt.py     情绪识别 prompt 模板
  emotion_examples.py   本地 few-shot 示例库
  emotion_retrieval.py  动态示例选择
  emotion_feedback.py   情绪反馈持久化
  safety.py             当前输入的安全提示判断

  memory.py             长期记忆协议、配置和 prompt 格式化
  local_memory.py       SQLite 本地记忆 provider
  memory_extractor.py   保守抽取长期记忆候选

  history.py            聊天历史、AI 反馈和重新生成记录
  profile.py            静态用户画像加载和格式化
  static/               前端页面、样式和交互脚本

data/
  examples/             评估脚本样例数据
  config/               本地 prompt 模板配置样例
  records/              运行时本地状态

scripts/
  evaluate_emotion_analysis.py  评估单个情绪分析结果文件
  evaluate_emotion_ablation.py  对比多组情绪分析结果

tests/                  单元测试和 Web 接口测试
```

## 配置

`load_config()` 的优先级是 CLI 参数、环境变量、默认值。Web 启动时调用 `load_config([])`，因此通常通过 `.env` 或环境变量配置。

### 聊天 LLM

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | 聊天 LLM provider。当前支持 `openai` 和 `deepseek`。 |
| `LLM_API_KEY` | 无 | 必填。也兼容旧变量 `OPENAI_API_KEY`。 |
| `LLM_MODEL` | `gpt-4o-mini` | 聊天模型。也兼容旧变量 `OPENAI_MODEL`。 |
| `LLM_BASE_URL` | 空 | OpenAI-compatible API base URL。也兼容旧变量 `OPENAI_BASE_URL`。 |
| `LLM_TEMPERATURE` | `0.7` | 聊天模型采样温度。也兼容旧变量 `OPENAI_TEMPERATURE`。 |

### 情绪 LLM

如果所有 `EMOTION_LLM_*` 都为空或未设置，情绪识别会复用聊天 LLM 实例。只要设置任一 `EMOTION_LLM_*`，项目会构建独立情绪 LLM，未设置的字段继承聊天 LLM。

| 变量 | 未设置时 |
| --- | --- |
| `EMOTION_LLM_PROVIDER` | 继承 `LLM_PROVIDER`。 |
| `EMOTION_LLM_API_KEY` | 继承 `LLM_API_KEY`。 |
| `EMOTION_LLM_MODEL` | 继承 `LLM_MODEL`。 |
| `EMOTION_LLM_BASE_URL` | 继承 `LLM_BASE_URL`。 |
| `EMOTION_LLM_TEMPERATURE` | 继承 `LLM_TEMPERATURE`。 |
| `EMOTION_INTERVAL` | 默认 `5`，必须是正整数。 |

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

### 长期记忆

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MEMORY_ENABLED` | `true` | 是否启用本地长期记忆；`0`、`false`、`no`、`off` 会关闭。 |
| `MEMORY_DB_PATH` | `data/records/memory.sqlite3` | SQLite 记忆数据库路径。 |
| `MEMORY_MAX_RESULTS` | `5` | 每轮最多注入多少条相关记忆；无效值回退到默认值。 |

### 长期记忆提炼

记忆提炼默认保持本地优先，不使用 Mem0、Qdrant 或托管向量库。应用会每隔几轮把最近一段 human/AI 对话窗口提炼为更稳定的长期记忆候选，再通过现有 SQLite memory provider 合并、去重和处理冲突。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MEMORY_CONSOLIDATION_ENABLED` | `true` | 是否启用周期性长期记忆提炼；关闭长期记忆时也会关闭提炼。 |
| `MEMORY_CONSOLIDATION_INTERVAL` | `5` | 每隔多少个用户回合尝试提炼一次。 |
| `MEMORY_CONSOLIDATION_WINDOW` | `12` | 每次最多查看最近多少条 human/AI 消息。 |
| `MEMORY_CONSOLIDATION_MODE` | `rules` | 当前支持本地规则模式；无效值回退为 `rules`。 |

### Prompt 模板

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PROMPT_CONFIG_PATH` | `data/config/prompts.json` | 本地 prompt 模板配置路径；文件缺失、JSON 损坏、字段为空或类型不对时使用内置默认 prompt。 |

可从 `data/config/prompts.example.json` 复制一份为 `data/config/prompts.json` 后修改。当前支持：

- `chat_system`：聊天 LLM 的 system prompt 主体。用户画像、长期记忆上下文和情绪上下文仍由程序追加/注入。
- `emotion_analysis`：情绪识别 prompt 模板。可用占位符包括 `{emotion_labels}`、`{example_block}`、`{likely_line}`、`{dialogue_context}`。如果模板格式错误，会回退到内置默认模板。

长期记忆分类固定为：

- `preference`：用户偏好。
- `profile`：稳定画像。
- `goal`：长期目标。
- `boundary`：明确约束。

注入聊天 prompt 的格式是：

```text
Relevant Long-term Memory:
- 用户希望回答使用中文。
- 用户希望项目保持本地优先，不引入第三方托管存储。
```

## 情绪识别

情绪识别模块会把最近对话上下文分类到固定情绪标签，并保存结构化状态：

```json
{
  "primary_emotion": "anxious",
  "confidence": 0.82,
  "secondary_emotions": ["hopeful"],
  "evidence": "I am worried about tomorrow's demo but I still want to try.",
  "reply_strategy": "Acknowledge the worry, encourage a small next step, and keep the reply calm.",
  "trajectory_note": "Anxiety is present, with some hopeful intent.",
  "safety_level": "normal"
}
```

LLM 也可以返回旧格式 `Emotion: anxious`。旧格式会被解析成最小结构化状态。

当前情绪分析使用三类上下文：

- 最近对话内容，最多按 `EMOTION_INTERVAL` 轮截取。
- 动态选择的本地 EICL/few-shot 示例。
- 最近 3 个成功识别的候选情绪。

情绪状态会在后续聊天中格式化为 `emotion_context`。如果当前输入触发安全提示，`safety.py` 会把状态标为 `supportive` 或 `crisis`，并把回复策略替换为安全引导语。

## 本地数据文件

默认运行时状态：

| 文件 | 内容 |
| --- | --- |
| `data/records/runtime.sqlite3` | 聊天历史、用户画像、AI 消息反馈、重新生成记录、情绪分析记录和情绪识别正确性反馈；用户画像保存在 `profile_entries` 表中。 |
| `data/records/memory.sqlite3` | 本地长期记忆；长期记忆仍保存在此数据库中。 |

运行时记录不再读取旧 JSON 文件，也不会自动迁移旧历史；需要保留的内容应直接写入本地 SQLite 数据库。`data/records/` 默认不提交到 Git。

## 评估脚本

评估单个情绪分析结果文件：

```bash
python scripts/evaluate_emotion_analysis.py \
  --analysis-file data/examples/dynamic_eicl_sample.json \
  --labels-file data/examples/emotion_labels_sample.json
```

评估默认运行时记录：

```bash
python scripts/evaluate_emotion_analysis.py \
  --analysis-file data/records/emotion_analysis.json \
  --labels-file data/examples/emotion_labels_sample.json
```

标注文件支持 JSON 数组或 JSONL。字段可以使用 `expected`、`emotion` 或 `label`：

```json
[
  {"turn_count": 5, "expected": "anxious"},
  {"turn_count": 10, "expected": "grateful"},
  {"turn_count": 15, "expected": "disappointed"}
]
```

匹配优先级：

1. `id` / `case_id`：稳定的固定消融用例身份。
2. `index`：第 N 条成功情绪分析记录。
3. `turn_count`：同一用户轮数。
4. `timestamp`：同一时间戳。
5. 无匹配字段时，按成功情绪分析记录顺序匹配。

输出包含样本数、正确数、accuracy、macro F1 和错误样例。

对比多组情绪分析结果：

```bash
python scripts/evaluate_emotion_ablation.py \
  --labels-file data/examples/emotion_labels_sample.json \
  --run static=data/examples/static_few_shot_sample.json \
  --run dynamic=data/examples/dynamic_eicl_sample.json \
  --markdown-file data/records/ablation.md \
  --csv-file data/records/ablation.csv
```

### 主应用情绪识别消融

第一阶段消融实验使用固定 JSONL 样本，比较完整情绪识别链路和几个受控变体：

- `full`：默认上下文窗口、动态示例、历史情绪先验。
- `no_dynamic_examples`：关闭动态示例检索，使用静态示例。
- `no_emotion_history`：不注入 previous/likely emotion。
- `short_context`：只使用 1 轮上下文。
- `zero_shot`：无 few-shot 示例，无历史情绪先验。

生成各组情绪分析结果：

```bash
python scripts/run_emotion_ablation.py \
  --dialogues-file data/examples/ablation_dialogues.jsonl \
  --output-dir data/records/ablation
```

汇总对比：

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

`summary.md` 可直接用于报告主表，`metrics.csv` 用于保存原始指标。

## 测试

运行完整测试：

```bash
pytest
```

如果 shell 找不到 `pytest`，使用虚拟环境里的解释器：

```bash
.venv/bin/python -m pytest -q
```

测试覆盖配置加载、LLM 适配、聊天服务编排、历史持久化、情绪解析、情绪反馈、长期记忆、评估脚本、README 示例和 FastAPI 接口。
