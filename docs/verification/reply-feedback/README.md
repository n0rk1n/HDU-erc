# 回复评价迁移验证

使用真实 FastAPI/SQLite 应用及离线模型，通过 Playwright Chromium 检查点赞、点踩、保存失败重试、刷新恢复、多气泡共用评价、移动端、切换用户和失败回复不可评价。

运行测试服务：`PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m tests.ui.preview_bubbles`。

另一个终端执行 `node tests/ui/feedback_browser.cjs`；可用 `PLAYWRIGHT_MODULE` 和 `CHROMIUM_EXECUTABLE` 指定现有安装。

`desktop.png`、`mobile.png` 为实际浏览器截图。测试使用工作树内 `data/bubbles-ui.sqlite3`，不调用外部模型。

最终验证：新版 `438 passed, 1 skipped`（跳过真实模型测试）；删除旧实现后归档 `515 passed`；两套 JavaScript 语法检查与 `git diff --check` 通过。两套 pytest 均有同一条现有 Starlette/AnyIO 弃用警告。
