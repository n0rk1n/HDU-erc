# 情绪识别展示验收记录

日期：2026-09-07。基线：`main` / `94b3be2`。任务分支：`codex/emotion-progress-ui-20260907`。

## 已实现

- 助手消息上方显示可折叠的处理卡片，后端按真实节点进度推送更新；结束后折叠，摘要保留情绪和模型置信度。
- 实际识别成功时展示情绪中文标签、置信度和简短依据；跳过和失败分别提示。准备失败不声称调用过模型。
- 右上角显示当前用户最近一次成功识别的情绪和时间；不受可见消息分页影响，不把旧结果标为本轮识别。
- 历史恢复、请求重放、识别失败后继续回答、切换用户、断连后台继续处理均保留原语义。
- 消息和识别卡片在独立滚动区中展示，输入框不遮挡内容；跟随新发送的消息，用户上翻时不因流式文本强制跳到底部。
- 白名单只公开业务步骤和情绪摘要；不公开模型原始 reasoning、Prompt、回复策略、轨迹、供应商错误或密钥。不增加数据库表。
- 旧版仅有情绪分析、没有入口判定记录时，保留结果并省略不存在的判定步骤。

## 自动化验证

工作目录：`/private/tmp/HDU-erc-emotion-ui-20260907`。

```bash
PYTHON_DOTENV_DISABLED=1 /Users/oriki/Database/HDU-erc/.venv/bin/python -m pytest -q --tb=short
node --check chatbot/static/app.js
node --check chatbot/static/presentation.js
node --check tests/ui/emotion_browser.cjs
git diff --check
```

结果：**327 passed, 1 skipped**；JavaScript 语法与 diff 检查通过。

- 基线为 321 passed, 1 skipped；新增 6 项后端集成测试，并扩充原有前端行为测试。
- 新增测试先验证缺失行为导致失败，再实现通过；手机输入框遮挡和旧版分析记录的误判均有先失败后通过的回归验证。
- 1 项跳过为默认关闭的真实模型测试；仍有既存的 Starlette/AnyIO 弃用警告。

## 浏览器验收

使用独立离线服务 `tests.ui.preview_emotion`，真实 FastAPI、LangGraph、SQLite 和浏览器，通过确定性离线模型模拟耗时和失败。仅使用 `data/ui-preview.sqlite3`；没有访问原有聊天数据库，也未请求外部模型。

```bash
PYTHON_DOTENV_DISABLED=1 /Users/oriki/Database/HDU-erc/.venv/bin/python -m tests.ui.preview_emotion
```

另一个终端：

```bash
PLAYWRIGHT_MODULE=/Users/oriki/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright \
CHROMIUM_EXECUTABLE=/Users/oriki/Library/Caches/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-mac-arm64/chrome-headless-shell \
node tests/ui/emotion_browser.cjs
```

最终代码验收通过：

| 场景 | 结果 |
| --- | --- |
| 识别进行中 | 真实进度出现；未返回结果前右上角保持尚未识别 |
| 识别成功 | 摘要、展开详情及右上角一致；完成后自动折叠 |
| 本轮跳过 | 标明未调用，不展示新的识别结果 |
| 识别失败 | 保留最近成功情绪，明确本轮失败，聊天继续完成 |
| 刷新重进 | 从服务端历史恢复情绪与卡片 |
| 390px / 320px 手机宽度 | 无横向溢出，聊天区与输入框不重叠 |
| 用户切换 | 清空上一用户状态；忽略延迟结果 |
| 断开显示后重新进入 | 后台完成的消息与结果可恢复 |
| 浏览器异常 | 无 pageerror |

截图已人工检查；下列情绪与回复均来自离线验收模型，不代表真实模型识别质量。

- [桌面处理过程](desktop-processing.png)
- [桌面识别结果](desktop-result.png)
- [手机识别结果](mobile-result.png)

## 提交边界

原目录与 `main` 保持不变；本次仅在任务工作树实现并提交。不合并、不推送、不替换现有运行服务。没有启动额外子代理；最终差异由本任务直接审查。

## 2026-09-07 配色调整

根据页面一致性反馈，将分析卡片、右上角情绪状态、情绪标签、头像和进度指示改为直接复用页面的 `--surface`、`--surface-soft`、`--border`、`--text`、`--text-muted` 与 `--accent`，移除单独的浅蓝配色。完成和失败仍使用小范围语义提示色。本次只修改样式，沿用同一套交互流程。

本次重新运行 `tests/ui/emotion_browser.cjs`，全部浏览器场景通过；桌面和手机截图已更新并人工检查，`git diff --check` 通过。此次为纯样式调整，未重复运行 Python 全量测试。
