# AI 消息点赞点踩设计

## 背景

当前 Web 聊天页面通过 `/api/chat/stream` 使用 SSE 流式生成回复，并通过 `/api/session` 加载最近历史消息。聊天历史统一保存在 `data/records/chat_history.json`，历史记录主要包含 `role`、`content` 和 `timestamp`。

本次功能目标是在 Web 页面上给大语言模型每次新生成的内容添加点赞和点踩标记。用户点击后，系统记录用户对这条生成内容的评价；已经评价过的内容不再显示按钮。

## 范围

- 只支持新生成的 AI 回复评价，不对已有历史 AI 回复做迁移。

- 评价直接写回 `chat_history.json` 中对应的 AI 消息记录。

- 每条可评价 AI 消息只能评价一次，首次评价后不允许覆盖。

- 前端只对有稳定 `id` 且 `feedback` 为空的 AI 消息显示点赞和点踩按钮。

## 架构

新生成的 AI 消息写入 `chat_history.json` 时附带稳定 `id`，例如 `ai_...`，并预留 `feedback` 字段，初始值为 `null`。历史加载接口 `/api/session` 会把这些字段一起返回给前端；旧 AI 历史因为没有 `id`，前端不会显示点赞和点踩按钮。

后端新增评价接口：

```http
POST /api/messages/{message_id}/feedback
```

请求体：

```json
{
  "feedback": "like"
}
```

`feedback` 只允许 `"like"` 或 `"dislike"`。接口读取 `chat_history.json`，找到对应 AI 消息；如果消息未评价，则写入评价并保存；如果已评价，则返回已评价状态，前端保持按钮隐藏。

前端把 AI 消息渲染拆成“气泡内容 + 评价操作区”。只有满足以下条件时显示按钮：消息角色是 `ai`、有 `id`、`feedback` 为空。用户点击后，按钮先禁用，接口成功后移除按钮；接口失败则恢复按钮并显示轻量错误提示。

## 数据流

发送消息时，现有 SSE 流程保持不变：`user_message` 事件先渲染用户消息，`token` 事件持续追加 AI 内容。

生成完成后，`ChatService.stream_reply()` 在完整答案生成成功后保存 AI 消息，并拿到这条消息的 `id`。随后 `done` 事件返回：

```json
{
  "content": "完整回复",
  "message_id": "ai_..."
}
```

前端收到 `done` 后，把当前正在流式渲染的 AI 气泡绑定这个 `message_id`，再显示点赞和点踩按钮。

刷新页面或重新打开页面时，`/api/session` 返回最近消息。新 AI 消息如果带 `id` 且没有 `feedback`，前端显示按钮；如果 `feedback` 已经是 `like` 或 `dislike`，前端只显示内容，不再显示按钮。旧 AI 消息没有 `id`，因此只显示内容。

用户点击点赞或点踩时，前端调用 `POST /api/messages/{message_id}/feedback`。后端校验评价值，成功写入后返回更新后的 `message_id` 和 `feedback`。前端收到成功响应后更新本地状态，并移除该消息的评价按钮。

## 错误处理

- 消息不存在时返回 `404`。

- 找到的记录不是 AI 消息时返回 `400`。

- `feedback` 不是 `like` 或 `dislike` 时返回 `422` 或 `400`。

- 消息已经有评价时不覆盖旧值，返回明确状态，例如 `{"status": "already_rated", "feedback": "like"}`。

- 保存历史文件失败时返回 `500`，前端恢复按钮可点击，并在消息旁显示“评价保存失败”。

- SSE 生成回复失败时，不写入完整 AI 历史，也不返回 `message_id`，前端不显示评价按钮。

## 测试

后端测试覆盖：

- AI 消息写入时生成 `id`，`feedback` 初始为 `None`；human 消息保持现有字段。

- `/api/session` 返回消息时保留新 AI 消息的 `id` 和 `feedback` 字段，旧消息仍正常返回。

- `POST /api/messages/{message_id}/feedback` 可以成功写入 `like` 和 `dislike`。

- 对不存在消息、非 AI 消息、已经评价过的消息、非法 `feedback` 做断言。

- SSE `done` 事件包含 `message_id`。

前端测试沿用当前 `tests/test_web.py` 中的 Node VM 风格，覆盖：

- 历史加载时，未评价的新 AI 消息会渲染评价按钮。

- 点击评价按钮后调用正确接口，成功后按钮隐藏。

- 已经评价或旧 AI 消息不显示按钮。

## 非目标

- 不做旧历史消息迁移。

- 不支持修改或撤销评价。

- 不新增点踩原因、统计面板或管理员视图。
