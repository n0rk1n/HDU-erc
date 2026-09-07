# GLM-5.3 模型切换验证

验证日期：2026-09-07。目标分支：`main`；任务分支：`codex/glm53-model-20260907`。

## 配置与实现

- 任务工作树的 `.env` 使用用户提供的北京百炼业务空间接口和密钥，文件权限为 `0600`，由 Git 忽略。共享模型为 `ZHIPU/GLM-5.3`，三个角色均继承 `enabled` 和 `low`。
- 新增 `LLM_REASONING_EFFORT` 和三个角色对应的覆盖项，百炼 GLM-5.3 请求通过 `extra_body` 发送 `enable_thinking` / `reasoning_effort`；关闭思考会在本地明确报错。
- `.env.example` 使用占位密钥和业务空间 ID，提供 GLM-5.3 默认配置。情绪/门控上下文为 1048576 token，输出预留 8192 token，使用固定版本的官方 tokenizer 和纯文本消息模板。
- 保留 DeepSeek V4 的原有参数和 token 计数行为。没有变更图结构、业务接口和原工作区数据库。

参数与上下文窗口依据：[百炼 GLM 接口文档](https://help.aliyun.com/zh/model-studio/glm-zhipu)、[GLM-5.3 模型信息](https://help.aliyun.com/zh/model-studio/glm-5-3-by-zhipu)。资源来源和校验值见 [官方分词资源说明](../../chatbot/llm/vendor/glm53/README.md)。

## 离线验证

在任务工作树内复用原项目的 Python 3.12 依赖解释器执行，禁用 dotenv 以隔离真实密钥和模型配置：

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/oriki/Database/HDU-erc/.venv/bin/python -m pytest -q --tb=short
```

结果：`397 passed, 1 skipped, 1 warning in 8.77s`。跳过的是默认关闭的在线模型测试；警告是现有 Starlette/AnyIO 的弃用提示。

新增请求参数和非法配置测试先验证失败（9 failed），再验证实现通过；GLM token 计数及启动测试先验证不支持 GLM tokenizer（7 failed），适配后与 DeepSeek 计数测试共同通过（22 passed）。覆盖三个角色的参数继承和独立覆盖、禁止关闭思考、完整消息预算、历史 reasoning、拒绝未计数载荷以及应用启动。

## 真实接口与完整应用验证

使用用户提供的实际接口和密钥，非模拟模型：

- 简短流式连通性请求返回“连接成功”，`finish_reason=stop`。
- 使用 FastAPI `TestClient`、真实模型客户端和任务工作树内的临时 SQLite 数据库，创建测试用户并发送两轮简短对话。
- 两轮 SSE 均以 `done` 结束，助手消息均持久化为 `completed`，`finish_reason=stop`。
- 首轮情绪分析为 `completed`，输入 1004 token、输出 97 token、无错误；第二轮实际门控调用为 `completed`，无错误。
- 两条聊天消息、情绪分析和门控调用的审计模型均为 `ZHIPU/GLM-5.3`，参数均为 `{"enable_thinking": true, "reasoning_effort": "low"}`。
- 临时测试数据库在验证后移除，原工作区数据未用于调用或修改。

直接流式脚本退出时曾出现 `httpcore2` 异步生成器清理异常，发生在成功收到完整回复之后。检查栈位置在现有 OpenAI SDK / httpcore2 的响应清理路径；完整应用两轮调用及正常关闭没有复现。此处仅记录观察，不宣称依赖告警已修复，也未修改依赖版本。

## 首次验证时的生效范围

首次验证结果仅位于任务分支和工作树；当时未合并、未推送、未重启原服务。真实密钥保存在忽略文件 `.env`，不会随 Git 合并传播。合并时需同步目标运行目录的模型配置，重启后才会切换原服务。

## 与最新 main 的集成验证

用户随后授权合并并删除任务分支和工作树。此时 `main` 已包含多气泡回复提交 `0af8626`，因此先在任务工作树合入最新目标分支。配置文件的冲突同时保留 `chat_bubble_gap_ms` 与 `llm_reasoning_effort` 的字段、校验和环境变量加载，并保留聊天请求的 JSON 输出格式及流关闭逻辑。

- 合并内容的完整离线测试：`425 passed, 1 skipped, 1 warning in 9.64s`。
- `RUN_LIVE_LLM_TEST=1` 执行合并后的 `tests/app/test_live_llm.py`：`1 passed, 1 warning in 3.91s`。实际 GLM-5.3 回复经应用处理后成功完成并持久化。
- 两次警告均为已有 Starlette/AnyIO 弃用提示。
