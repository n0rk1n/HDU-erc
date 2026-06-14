# 情绪识别聊天机器人

这是一个本地 Web 聊天机器人项目。它使用 LLM 生成聊天回复，通过 SSE 在浏览器中流式显示；同时会周期性识别用户情绪，并用本地 SQLite 记住长期偏好、目标和约束。聊天历史、情绪分析记录、AI 消息反馈和长期记忆都保存在本地。

项目默认不使用 Mem0 Platform，也不依赖第三方托管存储。长期记忆使用 Python 标准库 `sqlite3`，不需要单独安装数据库服务。

## 快速开始

创建并激活 Python 虚拟环境，然后安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

从示例文件创建本地 `.env`：

```bash
cp .env.example .env
```

编辑 `.env`，填入聊天模型配置。DeepSeek 示例：

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
```

启动 Web 应用：

```bash
uvicorn chatbot.web:app
```

然后打开：

```text
http://127.0.0.1:8000
```

## 核心功能

- 提供 FastAPI Web 服务和免构建的 HTML/CSS/JavaScript 前端。
- 使用 SSE 将 AI 回复按 token 流式显示到页面。
- 持久化聊天历史，并在页面加载时恢复最近消息。
- 每隔固定用户轮数进行一次情绪识别。
- 情绪识别 prompt 支持 few-shot 示例和最近候选情绪。
- 将当前情绪和相关长期记忆注入聊天 prompt。
- 支持对 AI 消息进行点赞或点踩反馈。
- 使用本地 SQLite 保存长期记忆，不需要第三方托管存储。
- 提供情绪识别评估脚本，支持 accuracy、macro F1 和错误样例输出。

## 运行流程

应用启动时，`chatbot.web` 会构建一个本地聊天服务：

1. 从 `.env` 和环境变量读取 LLM、情绪识别和记忆配置。
2. 读取 `data/records/chat_history.json` 中的历史消息。
3. 读取可选的静态用户画像。
4. 构建聊天 LLM 和情绪识别 LLM。
5. 构建本地 memory provider，默认使用 `data/records/memory.sqlite3`。
6. 将历史消息恢复到 LangChain 会话历史中。
7. 构建聊天 chain，并创建 `ChatService`。

用户发送消息时：

1. 前端用 `EventSource` 连接 `/api/chat/stream`。
2. `ChatService` 写入用户消息。
3. 根据当前输入检索相关长期记忆，生成 `memory_context`。
4. 如果到达情绪识别间隔，调用情绪 LLM 并更新当前情绪。
5. 将 `memory_context` 和 `emotion_context` 注入聊天 prompt。
6. 聊天 LLM 流式生成回复，前端逐段渲染。
7. 完整 AI 回复写入历史，并生成可反馈的 message id。
8. 从用户消息中保守提取长期记忆候选，写入本地 SQLite。

记忆检索、记忆写入或情绪识别失败时，聊天流程会继续；失败信息只作为 warning 或 SSE 状态暴露。

## 项目结构

```text
chatbot/
  web.py              FastAPI 应用、路由、静态页面服务和 SSE 接口
  chat_service.py     单条消息的业务编排：历史、记忆、情绪分析和回复生成
  llm.py              LangChain prompt、chain 构建和会话历史管理
  llm_adapter.py      OpenAI-compatible 模型适配层
  config.py           聊天/情绪 LLM 配置加载

  emotion.py          情绪分析流程、输出解析和结果持久化
  emotion_prompt.py   情绪识别 prompt 模板和渲染逻辑
  emotion_examples.py 情绪识别 few-shot 示例库

  memory.py           记忆数据结构、provider 协议、配置和上下文格式化
  local_memory.py     SQLite 本地记忆 provider
  memory_extractor.py 保守提取长期记忆候选

  history.py          聊天历史持久化和 AI 消息反馈更新
  profile.py          静态用户画像加载和格式化
  static/
    index.html        浏览器聊天页面
    style.css         聊天界面样式
    app.js            会话加载、SSE 处理和反馈交互

data/
  records/
    chat_history.json       聊天消息记录
    emotion_analysis.json   情绪分析记录
    memory.sqlite3          本地长期记忆

scripts/
  evaluate_emotion_analysis.py  情绪识别结果评估脚本

tests/                单元测试和 Web 接口测试
```

## LLM 配置

项目有两类 LLM：

- **聊天 LLM**：负责生成用户看到的回复，由 `LLM_*` 变量配置，必须提供 API key。
- **情绪识别 LLM**：负责判断用户当前情绪，由 `EMOTION_LLM_*` 变量配置，可选。

如果没有设置 `EMOTION_LLM_MODEL`，情绪识别会复用聊天 LLM：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.7
EMOTION_INTERVAL=5
```

如果设置了 `EMOTION_LLM_MODEL`，项目会为情绪识别构建独立 LLM。未单独设置的情绪配置会继承聊天 LLM：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_chat_api_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.7

EMOTION_LLM_MODEL=gpt-4o-mini
EMOTION_LLM_PROVIDER=openai
EMOTION_LLM_API_KEY=your_emotion_api_key
EMOTION_LLM_BASE_URL=https://api.openai.com/v1
EMOTION_LLM_TEMPERATURE=0
EMOTION_INTERVAL=5
```

继承规则：

| 情绪变量 | 未设置时的行为 |
| --- | --- |
| `EMOTION_LLM_MODEL` | 不构建独立情绪 LLM，直接复用聊天 LLM。 |
| `EMOTION_LLM_PROVIDER` | 使用聊天 LLM 的 provider。 |
| `EMOTION_LLM_API_KEY` | 使用聊天 LLM 的 API key。 |
| `EMOTION_LLM_BASE_URL` | 使用聊天 LLM 的 base URL。 |
| `EMOTION_LLM_TEMPERATURE` | 使用聊天 LLM 的 temperature。 |

当前模型 provider 支持 `openai` 和 `deepseek`，底层走 OpenAI-compatible 接口。

## 本地长期记忆

长期记忆用于保存稳定信息，例如：

- `preference`：用户偏好，例如希望使用中文、回答简洁。
- `profile`：稳定画像，例如正在开发某个长期项目。
- `goal`：持续目标，例如希望项目保持本地优先。
- `boundary`：明确约束，例如不要使用第三方托管存储。

默认记忆文件：

```text
data/records/memory.sqlite3
```

相关环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MEMORY_ENABLED` | `true` | 是否启用本地长期记忆。 |
| `MEMORY_DB_PATH` | `data/records/memory.sqlite3` | SQLite 记忆文件路径。 |
| `MEMORY_MAX_RESULTS` | `5` | 每次回复最多注入多少条相关记忆。 |

聊天时，系统会先根据当前输入检索相关记忆，并注入 prompt：

```text
Relevant Long-term Memory:
- 用户希望回答使用中文。
- 用户希望项目保持本地优先，不引入第三方托管存储。
```

回复成功后，系统会从用户消息中保守提取可长期保存的信息。临时闲聊、一次性情绪和不明确的信息不会主动写入长期记忆。

这套机制不使用 Mem0 Platform，不引入云端向量数据库，也不会把记忆内容存到第三方托管存储。以后如果需要接入 Mem0 OSS，可以通过新的 `MemoryProvider` 实现替换本地 provider，而不改变聊天主流程。

## 情绪识别

情绪识别模块负责把最近对话上下文分类到固定情绪标签中。它的结果会写入 `data/records/emotion_analysis.json`，并在后续回复中作为 `emotion_context` 注入聊天 prompt。

当前情绪识别做了三层增强：

- **Few-shot 示例**：`emotion_examples.py` 维护少量带标签的对话示例，prompt 会渲染为 `Labeled examples`。
- **候选情绪提示**：`ChatService` 记录最近 3 个成功识别的情绪，并作为 `More likely emotion labels` 注入下一次情绪识别 prompt。
- **独立 prompt 构造**：`emotion_prompt.py` 负责模板、示例渲染和候选情绪归一化；`emotion.py` 只负责分析流程、解析和持久化。

模型必须输出：

```text
Emotion: label
```

如果输出无法解析为已知标签，本轮聊天仍会继续，错误会记录到情绪分析文件。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `LLM_PROVIDER` | 聊天 LLM provider。当前支持 `openai` 和 `deepseek`。 |
| `LLM_API_KEY` | 聊天 LLM API key。必须配置。 |
| `LLM_MODEL` | 聊天模型名称。未设置时默认使用 `gpt-4o-mini`。 |
| `LLM_BASE_URL` | 可选的 OpenAI-compatible API base URL。 |
| `LLM_TEMPERATURE` | 聊天模型采样温度。默认值为 `0.7`。 |
| `EMOTION_LLM_MODEL` | 情绪识别 LLM 模型名称。设置后启用独立情绪 LLM。 |
| `EMOTION_LLM_PROVIDER` | 情绪识别 LLM provider。未设置时继承聊天 LLM。 |
| `EMOTION_LLM_API_KEY` | 情绪识别 LLM API key。未设置时继承聊天 LLM。 |
| `EMOTION_LLM_BASE_URL` | 情绪识别 LLM base URL。未设置时继承聊天 LLM。 |
| `EMOTION_LLM_TEMPERATURE` | 情绪识别 LLM 采样温度。未设置时继承聊天 LLM。 |
| `EMOTION_INTERVAL` | 每 N 个用户回合进行一次情绪识别。默认值为 `5`。 |
| `MEMORY_ENABLED` | 是否启用本地长期记忆。默认值为 `true`。 |
| `MEMORY_DB_PATH` | SQLite 记忆文件路径。默认值为 `data/records/memory.sqlite3`。 |
| `MEMORY_MAX_RESULTS` | 每次回复最多注入多少条相关记忆。默认值为 `5`。 |

旧的 `OPENAI_*` 变量仍然兼容。当新的 `LLM_*` 变量不存在时，会使用旧变量作为 fallback。

## 本地数据文件

运行时数据默认保存在 `data/records/`：

- `chat_history.json`：保存 human 和 AI 消息。AI 消息包含生成的 `id` 和可为空的 `feedback` 字段。
- `emotion_analysis.json`：保存每次情绪识别的 prompt、模型输出、解析结果、成功标记和错误信息。
- `memory.sqlite3`：保存本地长期记忆，包括用户偏好、稳定画像、长期目标和明确约束。

这些文件属于本地应用状态。删除或备份它们可以重置对应状态。

## 情绪识别评估

使用评估脚本对比 `emotion_analysis.json` 和人工标注文件：

```bash
python scripts/evaluate_emotion_analysis.py --labels-file data/records/emotion_labels.json
```

指定分析文件：

```bash
python scripts/evaluate_emotion_analysis.py \
  --analysis-file data/records/emotion_analysis.json \
  --labels-file data/records/emotion_labels.json
```

标注文件支持 JSON 数组或 JSONL。推荐 JSON 格式：

```json
[
  {"turn_count": 5, "expected": "anxious"},
  {"turn_count": 10, "expected": "grateful"},
  {"turn_count": 15, "expected": "disappointed"}
]
```

匹配规则：

- 有 `turn_count` 时，优先匹配同一轮数的情绪分析记录。
- 有 `timestamp` 时，匹配同一时间戳的情绪分析记录。
- 有 `index` 时，匹配第 N 条成功的情绪分析记录。
- 都没有时，按成功情绪分析记录的顺序匹配。

标注字段可以使用 `expected`、`emotion` 或 `label`。输出示例：

```text
Samples: 12
Correct: 9
Accuracy: 75.00%
Macro F1: 70.31%
Errors: 3
- 10: expected=grateful predicted=content
```

## 测试

运行测试：

```bash
pytest
```

如果当前环境没有可用的 `pytest`，可以先做语法检查：

```bash
python3 -m py_compile chatbot/*.py scripts/evaluate_emotion_analysis.py
```

测试覆盖配置加载、历史持久化、情绪识别、长期记忆、LLM 适配层、聊天服务编排和 FastAPI 接口。
