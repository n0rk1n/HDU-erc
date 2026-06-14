# Regenerate Feedback Design

## Goal

Add a regenerate flow for AI replies so users can request a new answer after selecting why the original answer was unsatisfactory. The system must preserve enough context to later analyze what answer caused dissatisfaction and why the user regenerated it.

## User Experience

Each AI reply that can receive feedback also shows a `Regenerate` action.

When the user selects `Regenerate`, the UI asks for one reason from this fixed set:

- `不准确`
- `不完整`
- `没有理解我的问题`
- `语气不合适`
- `其他`

After the user chooses a reason, the app regenerates a reply for the same user message that originally produced the selected AI answer.

The old AI reply remains in history but is collapsed by default after the new answer is available. The new AI reply appears normally in the conversation. This keeps the current chat readable while preserving the original answer for analysis.

## Data Model

The existing chat history JSON remains the source of truth. AI messages keep their current fields:

```json
{
  "id": "ai_old",
  "role": "ai",
  "content": "old answer",
  "timestamp": "...",
  "feedback": null
}
```

When regeneration succeeds, the original AI message receives a `regeneration` object:

```json
{
  "reason": "不准确",
  "regenerated_message_id": "ai_new",
  "timestamp": "...",
  "original_user_message": "the user prompt that produced ai_old",
  "original_ai_content": "old answer"
}
```

The regenerated AI message is appended as a normal AI message with a pointer to the original:

```json
{
  "id": "ai_new",
  "role": "ai",
  "content": "new answer",
  "timestamp": "...",
  "feedback": null,
  "regenerated_from": "ai_old"
}
```

The original AI content is duplicated inside `regeneration.original_ai_content` intentionally. It captures the exact answer that triggered dissatisfaction even if future display or migration logic changes the message body.

## Backend Design

Add a history-layer operation that records regeneration metadata atomically:

- Validate that the original message id exists.
- Validate that the original message is an AI message.
- Find the nearest preceding human message in history; this is the prompt to regenerate from.
- Reject unsupported reasons.
- Append the new AI answer as a normal AI record.
- Mark the original AI record with the regeneration event.

Add a service-layer regeneration method that reuses the existing reply-generation path as much as practical:

- It receives the original AI message id and selected reason.
- It asks the history layer for the original user prompt and current regeneration eligibility.
- It generates a new answer from the original user prompt.
- It records the regeneration event and appends the new AI record.
- It returns the new message id and content.

The first implementation should support one regeneration event per original AI message. If a user wants another regeneration, they can regenerate the newest AI reply. This avoids ambiguous chains on one old message while still allowing multiple attempts through `regenerated_from`.

## API Design

Add a new endpoint:

```http
POST /api/messages/{message_id}/regenerate
Content-Type: application/json

{
  "reason": "不准确"
}
```

Successful response:

```json
{
  "status": "regenerated",
  "original_message_id": "ai_old",
  "message_id": "ai_new",
  "content": "new answer",
  "reason": "不准确"
}
```

Expected errors:

- `404` when the original message does not exist.
- `400` when the target is not an AI message.
- `400` when the reason is unsupported.
- `409` when the original message has already been regenerated.
- `500` when the new answer or history write cannot be completed.

## Frontend Design

The existing feedback controls are extended with a `Regenerate` action.

Interaction flow:

1. User clicks `Regenerate`.
2. The controls show the fixed reason choices.
3. User selects one reason.
4. Buttons are disabled while the request runs.
5. On success, the old reply is collapsed and the new reply is inserted after it.
6. On failure, the old reply stays expanded and an inline error appears.

Collapsed old replies should still preserve enough visible context for the user, for example a compact row with text such as `已重新生成：不准确` and an expand button.

The session loader must understand the new fields:

- Messages with `regeneration` are rendered collapsed by default.
- Messages with `regenerated_from` are rendered as normal AI replies.

## Error Handling

If regeneration fails before the new answer is generated, no history mutation should be recorded.

If answer generation succeeds but writing history fails, the API returns an error and the frontend keeps the original message unchanged. The initial implementation does not need a retry queue.

If the original user prompt cannot be found, the backend returns a `400` response explaining that the original prompt is unavailable.

## Testing

Add focused tests for:

- History rejects unsupported regeneration reasons.
- History rejects regeneration for missing or non-AI messages.
- History finds the preceding human message for an AI reply.
- Successful regeneration records the reason, original user prompt, original AI content, regenerated message id, and `regenerated_from`.
- A second regeneration of the same original AI message returns a conflict.
- The web endpoint returns the expected success payload.
- The web endpoint maps not found, non-AI, invalid reason, already regenerated, missing prompt, and write failure cases to the expected HTTP responses.
- The frontend renders `Regenerate`, shows reason choices, submits the selected reason, collapses the old answer, and inserts the regenerated answer.

## Non-Goals

- Do not replace or delete old AI replies.
- Do not merge regeneration with the existing Good/Bad feedback value.
- Do not support free-text dissatisfaction reasons in the first version.
- Do not build a full version-history UI beyond collapsing the old answer and linking the new one.
