# 聊天气泡验证记录

日期：2026-09-07。验证对象为 `codex/chat-bubbles-20260907` 工作树，未修改主工作区的业务数据库或运行服务。

- 自动化回归：`python -m pytest -q -p no:cacheprovider --tb=short`，409 passed、1 skipped；保留原有 Starlette/AnyIO 弃用警告。覆盖 JSON 任意分块、完整内容保留、首条先保存再推送、失败恢复、v3 数据迁移、历史/幂等重放、前端排队与切换用户。
- 浏览器：先运行 `python -m tests.ui.preview_bubbles`，再运行 `node tests/ui/bubbles_browser.cjs`（需 Playwright 与 Chromium，可通过脚本中的环境变量指定路径）。七类场景通过：首条提前展示、500 ms 间隔、每轮一个处理卡片、刷新、390/320 px 无横向溢出、部分失败、排队时切换用户。最后一次实测相邻气泡间隔约 1204/504 ms，前一间隔包含测试模型的 1200 ms 延迟。
- [桌面截图](desktop.png)和[手机截图](mobile.png)使用隔离浏览器与确定性测试模型；截图关闭入场动画以捕获稳定状态。测试中的情绪识别服务为替身，不能据此判断真实识别效果。
- 真实模型：`python -m tests.ui.check_live_bubbles --env-file /path/to/.env`，使用生产聊天适配器调用 `deepseek-v4-flash`。凭据仅在进程内读取，输入均为人工构造的闲聊、倾诉、详细代码解释，未发送真实聊天记录。[原始结果](live.json)中三项均成功解码，分别为 2/2/3 个气泡，首条约 709–775 ms；最后一项保留完整代码。这是三次格式、流式行为与篇幅抽样，不代表长期稳定性或回答事实质量评测。

实现保留一次模型调用、一轮一条助手记录及现有一问一答流程。聊天接口需要支持 `response_format` 的 JSON 对象模式；正常情况下显示 1～3 个气泡，额外内容合入第三条，不按字数截断。
