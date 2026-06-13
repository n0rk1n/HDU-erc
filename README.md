# 情绪识别聊天机器人

这是一个本地 Web 聊天机器人项目，核心能力是把 LLM 对话和周期性情绪识别结合起来。应用会提供一个简单的浏览器聊天页面，通过 Server-Sent Events 流式返回 AI 回复，同时把聊天历史、用户反馈和情绪分析结果保存到本地 JSON 文件中。

## 项目功能

- 提供基于 FastAPI 的 Web 服务和免构建的 HTML/CSS/JavaScript 前端。
- 通过 SSE 将 AI 回复按 token 流式显示到页面。
- 将聊天历史保存到本地，并在页面加载时恢复最近消息。
- 每隔固定轮数分析一次用户当前情绪。
- 情绪识别 prompt 会带少量 few-shot 示例，并优先参考最近的候选情绪。
- 将最近检测到的情绪注入聊天 prompt，让回复更贴近用户状态。
- 支持对 AI 消息进行点赞或点踩反馈。
- 提供情绪识别评估脚本，可用人工标注文件计算 accuracy、macro F1 和错误样例。
- 支持 OpenAI-compatible 聊天模型 provider，包括 OpenAI 和 DeepSeek。

## 项目结构

```text
chatbot/
  web.py              FastAPI 应用、路由、静态页面服务和 SSE 接口
  chat_service.py     单条消息的业务编排：历史、情绪分析和回复生成
  llm.py              LangChain prompt、chain 构建和会话历史管理
  llm_adapter.py      OpenAI-compatible 模型适配层
  emotion.py          情绪 prompt 构建、输出解析和分析结果持久化
  emotion_prompt.py   情绪识别 prompt 模板和渲染逻辑
  emotion_examples.py 情绪识别 few-shot 示例库
  history.py          聊天历史持久化和 AI 消息反馈更新
  profile.py          用户画像加载和格式化
  config.py           环境变量和 CLI 配置加载
  static/
    index.html        浏览器聊天页面
    style.css         聊天界面样式
    app.js            会话加载、SSE 处理和反馈交互

data/
  records/
    chat_history.json       聊天消息记录
    emotion_analysis.json   情绪分析记录

tests/                单元测试和 Web 接口测试

scripts/
  evaluate_emotion_analysis.py  情绪识别结果评估脚本
```

## 运行时流程

应用启动时，`chatbot.web` 会构建一个本地聊天服务：

1. 从 `.env` 和环境变量读取配置。
2. 读取已持久化的聊天历史。
3. 读取可选的用户画像。
4. 构建聊天 LLM 和情绪分析 LLM。
5. 将历史消息恢复到 LangChain 会话历史中。
6. 构建聊天 chain，并创建 `ChatService`。

用户发送消息时，整体流程如下：

1. 前端使用 `EventSource` 连接 `/api/chat/stream`。
2. `ChatService` 先把用户消息写入历史。
3. 如果到达情绪分析间隔，就调用情绪 LLM 判断当前用户情绪。
4. 最新情绪会被格式化为聊天 LLM 的上下文。
5. 聊天 LLM 开始流式生成回复，并逐段返回给浏览器。
6. 完整 AI 回复生成后，会保存到历史文件，并生成消息 id。
7. 前端根据消息 id 显示点赞和点踩按钮。

## 情绪识别增强

情绪识别部分参考了 EICL 的两个思路，但在 chatbot 中做成轻量运行时机制：

- **Few-shot 示例**：`chatbot/emotion_examples.py` 中维护少量带标签的对话示例。构建情绪识别 prompt 时，会把这些示例加入 `Labeled examples` 区域，让模型参考“相似对话应该对应什么情绪”。
- **候选情绪提示**：`ChatService` 会记录最近成功识别到的 3 个情绪。下一次情绪识别时，这些情绪会以 `More likely emotion labels` 的形式放进 prompt，提醒模型优先考虑连续对话中的情绪惯性和渐变。
- **Prompt 构造独立模块**：情绪识别 prompt 的模板、示例渲染和候选情绪归一化集中在 `chatbot/emotion_prompt.py`，`chatbot/emotion.py` 只负责情绪分析流程、输出解析和持久化。

这些增强只影响情绪识别 LLM 的 prompt，不会直接覆盖最终结果。模型仍然必须从固定的情绪标签集中输出一个格式为 `Emotion: label` 的结果，解析失败时会记录错误，但聊天回复仍会继续。

## 安装

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

然后编辑 `.env`，填入模型配置。例如使用 OpenAI-compatible 的 DeepSeek：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.7
EMOTION_INTERVAL=5
```

如果没有设置 `EMOTION_LLM_MODEL`，情绪分析会复用聊天 LLM。只有在希望情绪识别使用独立模型时，才需要配置 `EMOTION_LLM_*` 相关变量。

## 启动

启动 Web 应用：

```bash
uvicorn chatbot.web:app
```

然后在浏览器打开：

```text
http://127.0.0.1:8000
```

## LLM 配置关系

项目里有两类 LLM 配置：

- **聊天 LLM**：负责生成用户实际看到的聊天回复。它由 `LLM_*` 变量配置，是应用必须配置的模型。
- **情绪识别 LLM**：负责每隔固定轮数判断用户当前情绪。它由 `EMOTION_LLM_*` 变量配置，是可选的独立模型。

默认情况下，情绪识别 LLM 会复用聊天 LLM。也就是说，只配置 `LLM_*` 就可以完整运行项目：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.7
EMOTION_INTERVAL=5
```

只有当设置了 `EMOTION_LLM_MODEL` 时，项目才会为情绪识别构建一套独立 LLM 配置。没有单独设置的情绪配置项会从聊天 LLM 继承：

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

情绪识别模型的继承规则如下：

| 情绪变量 | 未设置时的行为 |
| --- | --- |
| `EMOTION_LLM_MODEL` | 不构建独立情绪 LLM，直接复用聊天 LLM。 |
| `EMOTION_LLM_PROVIDER` | 使用聊天 LLM 的 provider。 |
| `EMOTION_LLM_API_KEY` | 使用聊天 LLM 的 API key。 |
| `EMOTION_LLM_BASE_URL` | 使用聊天 LLM 的 base URL。 |
| `EMOTION_LLM_TEMPERATURE` | 使用聊天 LLM 的 temperature。 |

这样可以按需要选择两种模式：

- 简单模式：聊天和情绪识别共用一个模型，配置少，启动方便。
- 独立模式：聊天模型负责自然回复，情绪模型可以换成更便宜、更稳定或更适合分类的模型。

## 环境变量

完整环境变量如下：

| 变量 | 作用 |
| --- | --- |
| `LLM_PROVIDER` | 聊天 LLM 的 provider。当前支持 `openai` 和 `deepseek`。 |
| `LLM_API_KEY` | 聊天 LLM 的 API key。必须配置。 |
| `LLM_MODEL` | 聊天模型名称。未设置时默认使用 `gpt-4o-mini`。 |
| `LLM_BASE_URL` | 可选的 OpenAI-compatible API base URL。 |
| `LLM_TEMPERATURE` | 聊天模型采样温度。默认值为 `0.7`。 |
| `EMOTION_LLM_MODEL` | 情绪识别 LLM 的模型名称。只有设置它时才启用独立情绪 LLM。 |
| `EMOTION_LLM_PROVIDER` | 情绪识别 LLM 的 provider。未设置时继承聊天 LLM。 |
| `EMOTION_LLM_API_KEY` | 情绪识别 LLM 的 API key。未设置时继承聊天 LLM。 |
| `EMOTION_LLM_BASE_URL` | 情绪识别 LLM 的 base URL。未设置时继承聊天 LLM。 |
| `EMOTION_LLM_TEMPERATURE` | 情绪识别 LLM 的采样温度。未设置时继承聊天 LLM。 |
| `EMOTION_INTERVAL` | 每 N 个用户回合进行一次情绪分析。默认值为 `5`。 |

当新的 `LLM_*` 变量不存在时，项目仍然兼容旧的 `OPENAI_*` 变量。

## 数据文件

应用运行时数据保存在 `data/records/` 下：

- `chat_history.json` 保存 human 和 AI 消息。AI 消息会包含生成的 `id` 和可为空的 `feedback` 字段。
- `emotion_analysis.json` 保存每次情绪分析的 prompt、模型输出、解析得到的情绪、成功标记，以及失败时的错误信息。

这些文件属于本地应用状态。如果想重置对话，可以删除或备份这些文件。

## 情绪识别评估

项目提供了一个轻量评估脚本，用来把 `emotion_analysis.json` 中的预测结果和人工标注文件对比，输出 accuracy、macro F1 和错误样例：

```bash
python scripts/evaluate_emotion_analysis.py --labels-file data/records/emotion_labels.json
```

默认会读取 `data/records/emotion_analysis.json`。如果要指定其他分析文件，可以传入：

```bash
python scripts/evaluate_emotion_analysis.py \
  --analysis-file data/records/emotion_analysis.json \
  --labels-file data/records/emotion_labels.json
```

标注文件支持 JSON 数组或 JSONL。推荐 JSON 格式如下：

```json
[
  {"turn_count": 5, "expected": "anxious"},
  {"turn_count": 10, "expected": "grateful"},
  {"turn_count": 15, "expected": "disappointed"}
]
```

匹配规则如下：

- 如果标注项包含 `turn_count`，优先匹配同一轮数的情绪分析记录。
- 如果标注项包含 `timestamp`，会匹配同一时间戳的情绪分析记录。
- 如果标注项包含 `index`，会匹配第 N 条成功的情绪分析记录。
- 如果以上字段都没有，脚本会按成功情绪分析记录的顺序逐条匹配。

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

测试覆盖配置加载、历史持久化、情绪解析和分析行为、LLM 适配层、聊天服务编排以及 FastAPI 接口。
