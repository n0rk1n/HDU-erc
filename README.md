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

同一个 SQLite 文件包含三张业务表和 LangGraph SQLite Saver 的两张官方内部表：

| 表 | 职责 |
| --- | --- |
| `users` | 保存外部用户标识与内部自增用户 ID。 |
| `conversations` | 保存用户的默认对话、内部 `thread_id` 和对话状态。 |
| `messages` | 聊天与审计事实来源，保存正文、状态、供应商实际返回的 reasoning、Prompt、模型参数、Token、耗时和 trace。 |
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

`python3 --version` 必须显示 3.10、3.11、3.12 或更高版本。如果系统默认 `python3` 过旧，请把上述 `python3` 替换为实际的 Python 3.10+ 可执行文件；例如 Apple Silicon Homebrew 安装的 3.12 通常可用 `/opt/homebrew/bin/python3.12 -m venv .venv`。本项目当前本机验收使用 Python 3.12。

编辑 `.env`，至少填写模型服务的 `LLM_API_KEY`，并按供应商设置 `LLM_MODEL` 和可选的 `LLM_BASE_URL`。随后以单 worker 启动：

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

默认测试使用确定性离线模型 adapter，不需要密钥，也不会发起模型网络请求：

```bash
.venv/bin/python -m pytest -q
```

真实模型冒烟测试默认显示 `SKIPPED`。只有已经配置有效凭据并明确允许调用时才运行：

```bash
RUN_LIVE_LLM_TEST=1 .venv/bin/python -m pytest tests/app/test_live_llm.py -v
```

真实模型测试通过与完整离线测试通过是两个独立结论；不要用离线结果声称真实供应商调用成功。

## 数据、安全与部署边界

- 默认数据文件是 `data/chatbot.sqlite3`。备份前停止应用，并复制该文件及当时存在的 `-wal`、`-shm` 文件；更稳妥的方式是使用 SQLite 的在线备份能力。
- 用户标识只是数据分区键，不是身份认证。任何知道同一标识的人都可以访问同一份聊天数据；不要把本项目直接暴露到不可信网络。
- 聊天正文、供应商实际返回的 reasoning、Prompt 和 trace 会以明文保存在 SQLite 中，可能包含敏感内容。请限制文件权限、备份访问和保留周期；不要把运行数据库提交到 Git。
- 首期只支持一个应用进程和 `--workers 1`。进程内并发锁与 SQLite 方案不能作为多进程或生产集群的一致性保证。
- 当前未实现身份认证、情绪识别、多会话创建/切换 UI、跨会话长期记忆、reasoning/trace 调试界面或生产级多实例数据库。

## 旧版归档

旧的 emotion-aware chatbot 已冻结在 `archive/emotion-aware-chatbot-v1/`，对应基线提交 `86220ac52df1641fee051bae8e4e04fd09ff6e40`，共 166 个受跟踪文件。归档仅供追溯和独立运行；新版不导入其模块，也不与它共享 Python 环境或依赖。运行方法见归档内的 `ARCHIVE.md`。
