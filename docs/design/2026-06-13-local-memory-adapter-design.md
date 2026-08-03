# Local Memory Adapter Design

## Goal

Add a lightweight local long-term memory layer to the chatbot without introducing Mem0 Platform, cloud vector databases, or any third-party hosted storage. The memory layer should help the assistant remember stable user preferences, profile facts, ongoing goals, and explicit boundaries across conversations.

The first version should use Python's built-in SQLite support through `sqlite3`, storing data in the existing runtime data directory:

```text
data/records/memory.sqlite3
```

## Non-Goals

- Do not integrate Mem0 Platform.
- Do not add a vector database or embedding dependency in the first version.
- Do not send memory contents to a third-party storage service.
- Do not build a user-facing memory management UI yet.
- Do not replace `chat_history.json`; chat history remains the conversation log, while memory stores durable facts.

## Current Context

The project already has these persistence and context boundaries:

- `chatbot/history.py` stores chat messages in `data/records/chat_history.json`.
- `chatbot/emotion.py` stores emotion-analysis records in `data/records/emotion_analysis.json`.
- `chatbot/profile.py` loads a static user profile from `data/config/user_profile.json`.
- `chatbot/llm.py` builds the chat prompt and injects `emotion_context`.
- `chatbot/chat_service.py` orchestrates each message: history write, optional emotion analysis, LLM reply, and AI message write.

The memory layer should fit beside these boundaries rather than absorbing them. Static profile data remains in `profile.py`; dynamic long-term memory goes into the new local memory adapter.

## Proposed Architecture

Add three focused modules:

```text
chatbot/
  memory.py             Memory dataclass, MemoryProvider protocol, context formatting
  local_memory.py       SQLiteLocalMemoryProvider implementation
  memory_extractor.py   Conservative extraction of memory candidates from dialogue
```

`memory.py` defines the public interface. `ChatService` should depend on the `MemoryProvider` protocol, not directly on SQLite. This keeps a future Mem0 OSS adapter possible without changing chat orchestration.

`local_memory.py` owns all SQLite details: opening the database, creating the schema, inserting or updating memories, searching, and recording usage.

`memory_extractor.py` decides whether a turn contains durable information worth remembering. The first version should use conservative local rules rather than an LLM extractor.

## Data Model

SQLite file:

```text
data/records/memory.sqlite3
```

Table:

```text
memories
  id TEXT PRIMARY KEY
  content TEXT NOT NULL
  category TEXT NOT NULL
  source TEXT NOT NULL
  confidence REAL NOT NULL
  created_at TEXT NOT NULL
  updated_at TEXT NOT NULL
  last_used_at TEXT
  use_count INTEGER NOT NULL DEFAULT 0
```

Allowed categories:

- `preference`: user preferences, such as preferred language or response style.
- `profile`: stable user facts, such as long-running projects or work context.
- `goal`: ongoing user goals.
- `boundary`: explicit constraints or things the user does not want.

The provider should normalize content before comparison and avoid inserting obvious duplicates. If a similar memory already exists, update `updated_at`, `confidence`, and category/source when appropriate instead of adding another row.

## Memory Extraction

The first version should be conservative. Extract memory only from clear user statements, especially patterns like:

- "我喜欢..."
- "我希望..."
- "以后请..."
- "不要..."
- "我的...是..."
- "I like..."
- "I prefer..."
- "Please remember..."
- "Do not..."
- "My ... is ..."

Each chat turn may produce at most three memory candidates. The extractor should not store one-off feelings, transient requests, or sensitive information unless the user clearly phrases it as something to remember or a durable preference.

The assistant reply may be passed into the extractor for context, but new memories should be based primarily on the user's message.

## Retrieval And Prompt Injection

Before generating a reply, `ChatService` should query the memory provider with the current user message.

Retrieval strategy for the first version:

- Tokenize the current input with simple local rules.
- Match tokens against memory `content`.
- Score by keyword overlap, then by recent `updated_at`, then by `use_count`.
- Return at most 3 to 5 memories.

`memory.py` should format retrieved memories as:

```text
Relevant Long-term Memory:
- 用户希望回答使用中文。
- 用户希望项目保持本地优先，不引入第三方托管存储。
```

`chatbot/llm.py` should add a `{memory_context}` slot to the system message, near the static `User Profile` and `emotion_context` sections.

`ChatService._payload()` should expand from:

```python
{
    "input": message,
    "emotion_context": ...,
}
```

to:

```python
{
    "input": message,
    "emotion_context": ...,
    "memory_context": ...,
}
```

## Chat Flow

For each user message:

1. Append the user message to chat history as it does today.
2. Retrieve relevant local memories for the current message.
3. Run emotion analysis if the interval is due.
4. Invoke or stream the chat LLM with `memory_context` and `emotion_context`.
5. Append the AI reply to chat history.
6. Extract memory candidates from the user message and AI reply.
7. Insert or update local memories in SQLite.

Memory failures should never break the chat response. Retrieval failure should produce an empty memory context. Write failure should emit a warning and leave the conversation flow intact.

## Configuration

Add environment variables with local defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEMORY_ENABLED` | `true` | Enable or disable local memory. |
| `MEMORY_DB_PATH` | `data/records/memory.sqlite3` | SQLite memory file path. |
| `MEMORY_MAX_RESULTS` | `5` | Maximum memories injected into one reply. |

No new external service credentials are required.

## Error Handling

- If SQLite database file does not exist, create it and initialize the schema.
- If the directory does not exist, create it.
- If retrieval fails, log a warning and continue with no memory context.
- If memory insertion/update fails, log a warning and continue.
- If the database is locked, skip the memory operation for that turn rather than blocking chat streaming.

## Testing

Add focused tests for:

- Schema creation in a temporary SQLite file.
- Insert and retrieve memory records.
- Duplicate or similar memory update behavior.
- Conservative extraction rules for preferences, goals, and boundaries.
- `ChatService` injecting memory context into the LLM payload.
- Chat continues when memory retrieval or write fails.
- `MEMORY_ENABLED=false` disables memory retrieval and writes.

The test suite should avoid relying on global `data/records/memory.sqlite3`; use temporary paths.

## README Updates

Document that:

- The memory layer uses local SQLite through Python's standard library.
- No separate database installation is required.
- The default file is `data/records/memory.sqlite3`.
- This does not use Mem0 Platform and does not introduce third-party hosted storage.
- A future Mem0 OSS adapter could implement the same `MemoryProvider` interface if needed.

## Future Extension

This design intentionally leaves room for future adapters:

- `SQLiteLocalMemoryProvider` for default local storage.
- Optional `Mem0OssMemoryProvider` later, if the project wants Mem0 OSS while preserving local/self-hosted control.
- Optional embedding-based retrieval later, if the project accepts the dependency and privacy trade-offs.

The first version should stay local, simple, and explainable.
