# 多用户 LangGraph 聊天机器人

这是一个首期本地版聊天应用：FastAPI 提供用户解析、消息历史和 SSE 流式消息三个公开 API，LangGraph 编排一次对话回复，SQLite 同时保存业务事实和官方 Checkpointer 状态。浏览器目前只有一个默认对话界面。

## 功能与架构

- 以用户标识解析或复用用户，并自动创建一个默认对话。
- 同一用户的已完成消息会进入下一轮模型 Prompt；不同用户的历史、Prompt 和 checkpoint 相互隔离。
- 流式回复支持 request ID 幂等重放；进程重启会保留已完成消息和正常 checkpoint，并把遗留中的回复标记为 `failed/process_interrupted`。
- 模型调用通过 OpenAI-compatible 接口完成；普通历史 API 和前端不会返回内部 `thread_id`、Prompt、reasoning 或 trace。

一次请求的主要路径如下：

```text
浏览器 -> FastAPI -> TurnCoordinator -> LangGraph -> 模型适配器
                    |                    |
                    +------ SQLite <-----+
```

同一个 SQLite 文件包含四张业务表和 LangGraph SQLite Saver 的两张官方内部表：

| 表 | 职责 |
| --- | --- |
| `users` | 保存外部用户标识与内部自增用户 ID。 |
| `conversations` | 保存用户的默认对话、内部 `thread_id` 和对话状态。 |
| `messages` | 聊天与审计事实来源，保存正文、状态、供应商实际返回的 reasoning、Prompt、模型参数、Token、耗时和 trace。 |
| `emotion_analyses` | 每条新用户输入的情绪分析记录：完整请求快照、配置、预算裁剪、原始输出、解析结果及失败诊断。 |
| `checkpoints` | 官方 Saver 保存的 LangGraph checkpoint；业务 API 不解析其 BLOB。 |
| `writes` | 官方 Saver 保存的中间写入；由 LangGraph 管理。 |

`reasoning_content` 只保存供应商确实返回的 reasoning；供应商没有返回时保持 `NULL`。它不会显示在当前前端或普通历史 API 中。

## 安装与启动

需要 Python 3.10 或更高版本。先确认准备使用的解释器版本，再创建虚拟环境：

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

`python3 --version` 必须显示 3.10、3.11、3.12 或更高版本。如果系统默认 `python3` 过旧，请把上述 `python3` 替换为实际的 Python 3.10+ 可执行文件；例如 Apple Silicon Homebrew 安装的 3.12 通常可用 `/opt/homebrew/bin/python3.12 -m venv .venv`。本次情绪功能离线验收使用 Python 3.14。

编辑 `.env`，至少填写模型服务的 `LLM_API_KEY`，并按供应商设置 `LLM_MODEL` 和可选的 `LLM_BASE_URL`。同时必须填写下述情绪模型的 `EMOTION_CONTEXT_TOKENS` 和 `EMOTION_TOKENIZER_MODEL`。随后以单 worker 启动：

```bash
.venv/bin/uvicorn chatbot.web:app --workers 1 --no-access-log
```

浏览器访问 <http://127.0.0.1:8000>。

`chatbot.main:app` 是等价的兼容入口；如果需要工厂模式，也可运行 `.venv/bin/uvicorn chatbot.web:create_app --factory --workers 1 --no-access-log`。

## 配置

`.env.example` 只包含占位值：

- `LLM_API_KEY`：模型服务密钥，必填；不要提交真实值。
- `LLM_MODEL`：供应商模型名。
- `LLM_BASE_URL`：OpenAI-compatible 服务地址；官方默认地址可留空。
- `LLM_TEMPERATURE`：采样温度，范围 0 到 2。
- `LLM_TIMEOUT_SECONDS`：模型调用和关闭等待秒数，必须大于 0。
- `CHAT_CONTEXT_MESSAGE_LIMIT`：每轮读取的最近已完成消息数，范围 1 到 200。
- `SQLITE_DB_PATH`：SQLite 文件路径，默认 `data/chatbot.sqlite3`。

修改 `.env` 后需要重启应用。

## 测试

默认测试使用确定性离线模型 adapter，不需要密钥，也不会发起模型网络请求。测试需要 Node.js 在 PATH 中，以运行现有前端脚本测试；禁用 dotenv 自动查找，避免 worktree 测试读到父目录的真实配置：

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q
```

真实模型冒烟测试默认显示 `SKIPPED`。只有已经配置有效凭据并明确允许调用时才运行：

```bash
RUN_LIVE_LLM_TEST=1 .venv/bin/python -m pytest tests/app/test_live_llm.py -v
```

真实模型测试通过与完整离线测试通过是两个独立结论；不要用离线结果声称真实供应商调用成功。

## 逐轮情绪分析

主图按 `prepare_turn → analyze_emotion → generate_response → finalize_turn` 执行；情绪节点内部为独立的准备、调用、保存子图。每个新 request_id 分析一次，相同文字的新请求仍会重新分析；重复请求不重复调用模型。

- `config/emotion_labels.json`：`"情绪": "描述"`，默认包括原有 32 类以及 `neutral` 和 `no_emotion`。
- `config/emotion_families.json`：`"情绪": "family"`，例如 sad 和 lonely 都属于 sadness_loss。family 由程序查表派生。
- `config/emotion_examples.json`：带标签的 few-shot 示例库，动态检索最多 4 条，并保存选择理由和分数。

`neutral` 指明确表达中性状态，例如“既不高兴，也不难过”；`no_emotion` 指未表达情绪，例如“转换这个文件”。二者都不是失败标记。配置加载拒绝重复键、空值、未知示例标签及不完整 family 映射；修改后重启生效，每次调用保存当时配置快照及哈希。

情绪模型连接参数 `EMOTION_LLM_API_KEY/MODEL/BASE_URL` 可分别覆盖聊天配置；默认温度 0，超时继承聊天超时，自动重试次数为 0。`EMOTION_LLM_TIMEOUT_SECONDS` 可独立调整。两次串行调用的关闭等待预算相加。

必须显式配置 `EMOTION_CONTEXT_TOKENS`（所部署模型的实际上下文窗口）和 `EMOTION_TOKENIZER_MODEL`（与服务匹配的分词器及聊天模板模型）。目前使用安装版 LangChain/tiktoken 支持的文本聊天计数接口，拒绝未知模型的静默回退。对不受支持的私有模型，需要先实现匹配的 TokenCounter 并通过 EmotionRuntime 注入，不能随便填一个 OpenAI 模型名替代。

历史候选来自当前会话，截至本条用户输入。按 request_id 配对为完整 human→assistant 轮次，排除未完成或失败的历史轮次；API 发送时映射为 user/assistant。优先保留最近轮次和完整的当前 human 消息，超预算删除最早整轮，不截断单条、不跳选更早短轮，也不删除数据库原始历史。

聊天消息（含当前输入和消息开销）不得超过上下文 C 的 60%；完整提示词还必须满足 `输入 token + 输出预留 + 安全余量 ≤ C`。`EMOTION_OUTPUT_TOKENS` 默认 1024，`EMOTION_SAFETY_TOKENS` 默认 256。本地 token 数是与所选模板关联的估计，安全余量用于协议/计数差异；不能保证任意供应商与本地估计完全相同。聊天回复自身继续使用 CHAT_CONTEXT_MESSAGE_LIMIT，不受此裁剪器改写。

当前输入单独超预算时，不截断它；保存 context_budget_exceeded，跳过情绪模型并继续普通回复。其他模型、解析和校验失败同样先保存可获取的信息再继续回复，不复用旧情绪。数据库持久化本身失败时本轮报基础设施错误。启动恢复把未完成分析标为 process_interrupted，保留已提交快照；不会自动重试旧分析。

`.env.example` 已将 `EMOTION_LABELS_PATH`、`EMOTION_FAMILIES_PATH`、`EMOTION_EXAMPLES_PATH` 指向 `data/config/` 下对应的三个 JSON 文件，提供完整的 34 类情绪、family 映射和动态检索示例，可直接编辑并在修改后重启。使用这些相对路径时，请从项目根目录启动，或改填绝对路径。变量留空时仍使用上述 `config/` 内置配置，其路径相对于项目模块位置，不依赖启动目录。

数据库启动时从 schema v1 增量升级到 v2，保留原消息和 checkpoint，不自动回填旧消息分析。查询每轮识别及失败信息：

```sql
SELECT user_message_id, status,
       json_extract(result_json, '$.primary_emotion') AS emotion,
       json_extract(result_json, '$.primary_family') AS family,
       json_extract(error_json, '$.stage') AS failure_stage,
       json_extract(error_json, '$.message') AS failure_message,
       input_tokens, output_tokens, latency_ms
FROM emotion_analyses ORDER BY created_at, id;
```

`snapshot_json` 保存实际入模提示词、human/assistant 历史、标签和 family、示例、有效参数、token 计数和裁剪 ID；`raw_output`、`result_json`、`response_metadata_json`、`error_json` 分别保留原始输出、校验结果、服务元数据及错误。错误诊断包含可获取的阶段、类型、状态码、服务端请求 ID 和错误正文；未知字段为 NULL。连接凭据脱敏。reasoning_content 只记录模型实际返回的内容。审计信息不公开到当前普通历史 API 或前端。

## 数据、安全与部署边界

- 默认数据文件是 `data/chatbot.sqlite3`。备份前停止应用，并复制该文件及当时存在的 `-wal`、`-shm` 文件；更稳妥的方式是使用 SQLite 的在线备份能力。
- 用户标识只是数据分区键，不是身份认证。任何知道同一标识的人都可以访问同一份聊天数据；不要把本项目直接暴露到不可信网络。
- 聊天正文、供应商实际返回的 reasoning、Prompt 和 trace 会以明文保存在 SQLite 中，可能包含敏感内容。请限制文件权限、备份访问和保留周期；不要把运行数据库提交到 Git。
- 首期只支持一个应用进程和 `--workers 1`。进程内并发锁与 SQLite 方案不能作为多进程或生产集群的一致性保证。
- 当前未实现身份认证、多会话创建/切换 UI、跨会话长期记忆、reasoning/trace 调试界面或生产级多实例数据库。

## 旧版归档

旧的 emotion-aware chatbot 已冻结在 `archive/emotion-aware-chatbot-v1/`，对应基线提交 `86220ac52df1641fee051bae8e4e04fd09ff6e40`，共 166 个受跟踪文件。归档仅供追溯和独立运行；新版不导入其模块，也不与它共享 Python 环境或依赖。运行方法见归档内的 `ARCHIVE.md`。
