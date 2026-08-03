# Emotional Memory Consolidation Design

## Goal

Improve the chatbot's long-term memory for emotional companionship by turning recent conversation windows into durable, useful context while preserving the current local-first architecture.

This design incorporates the article's non-Mem0 ideas:

- short-term memory as recent exact dialogue
- long-term memory as distilled key information
- periodic consolidation instead of unbounded context growth
- memory categories and metadata
- dynamic updates when newer information supersedes older memory
- optional question/context rewriting before memory retrieval

It does not introduce Mem0, Qdrant, hosted vector storage, or mandatory extra LLM calls on every chat turn.

## Current State

The project already has the main building blocks:

- `chatbot.llm` keeps short-term session continuity through LangChain chat history.
- `chatbot.history` persists the full chat log in `data/records/chat_history.json`.
- `chatbot.chat_service` retrieves long-term memory before generating a reply and writes new memory candidates after the reply.
- `chatbot.memory_extractor` conservatively extracts explicit user preferences, profile facts, and boundaries from a single user message.
- `chatbot.local_memory` stores local long-term memories in SQLite, handles duplicates and conflicts, and ranks retrieval using lexical relevance, category priority, confidence, recency, and usage.
- `chatbot.emotion` and `chatbot.emotion_state` provide structured emotional context that is already injected into the reply prompt.

The current memory flow is safe and local, but it mostly remembers explicit instructions. It does not yet summarize repeated emotional patterns, ongoing stressors, preferred support style, or multi-turn context that only becomes meaningful across several turns.

## Non-Goals

This work will not:

- replace SQLite with Mem0 or a vector database
- add a hosted memory service
- require network access beyond the existing configured LLM provider
- add a memory management UI
- store every transient feeling as a durable memory
- make memory consolidation block chat generation
- make live LLM calls necessary for unit tests

## Recommended Approach

Keep the current immediate local memory path and add a separate consolidation path.

Immediate memory remains rule-based and cheap:

1. User sends a message.
2. Chat retrieves relevant local memories.
3. Chat generates the assistant reply.
4. The extractor stores explicit durable statements from the user message.

Consolidated memory runs only after enough new conversation has accumulated:

1. Track how many user turns have happened since the last consolidation.
2. When the interval is reached, build a bounded window from recent human and AI messages.
3. Produce durable memory candidates from that window.
4. Merge candidates through the existing SQLite provider so duplicate and conflict rules still apply.
5. Record consolidation metadata so the same window is not repeatedly summarized.

The first implementation should keep consolidation rule-based by default and make LLM-based consolidation an optional later capability behind configuration.

## Configuration

Add memory consolidation settings beside the existing memory config:

```env
MEMORY_CONSOLIDATION_ENABLED=true
MEMORY_CONSOLIDATION_INTERVAL=5
MEMORY_CONSOLIDATION_WINDOW=12
MEMORY_CONSOLIDATION_MODE=rules
```

Meanings:

- `MEMORY_CONSOLIDATION_ENABLED`: enables or disables periodic consolidation.
- `MEMORY_CONSOLIDATION_INTERVAL`: number of user turns between consolidation attempts.
- `MEMORY_CONSOLIDATION_WINDOW`: maximum recent human/AI messages considered by one consolidation pass.
- `MEMORY_CONSOLIDATION_MODE`: `rules` for the first version; `llm` can be added later without changing the chat service contract.

Invalid values should fall back to safe defaults. Disabling long-term memory should also disable consolidation.

## Memory Categories

Keep the existing public categories for compatibility:

- `preference`
- `profile`
- `goal`
- `boundary`

Add optional internal metadata so emotional memories can be more specific without expanding the public provider contract too early:

```json
{
  "kind": "support_preference",
  "emotion": "anxious",
  "evidence_turns": ["..."],
  "consolidated_from": "chat_window",
  "consolidation_version": 1
}
```

Recommended internal `kind` values:

- `support_preference`: how the user wants to be supported or comforted
- `stress_context`: recurring stress source or life context
- `emotional_pattern`: repeated emotional pattern or trigger
- `ongoing_goal`: continuing task, project, or intention
- `communication_boundary`: explicit response style or support boundary

Map these to existing categories:

- `support_preference` -> `preference`
- `stress_context` -> `profile`
- `emotional_pattern` -> `profile`
- `ongoing_goal` -> `goal`
- `communication_boundary` -> `boundary`

This keeps prompt formatting and `MemoryProvider` stable while allowing better future ranking and filtering.

## Consolidation Rules

The default `rules` mode should be conservative and explainable.

It should create candidates only when the recent window contains durable signals, such as:

- repeated statements about the same stressor
- explicit phrases like "以后这样安慰我", "不要劝我", "我只是想被听见", "I just need you to listen"
- ongoing goals or projects mentioned across multiple turns
- stable communication preferences that affect future replies
- clear boundaries around advice, tone, language, or emotional support

It should avoid storing:

- one-off emotions like "今天有点累" unless repeated or framed as a pattern
- sensitive facts unless the user clearly presents them as durable context
- assistant guesses that the user did not confirm
- diagnostic labels or clinical conclusions
- crisis details beyond what is necessary for safe supportive behavior

Consolidation candidates should use cautious wording:

- "用户在最近对话中多次提到..."
- "用户倾向于..."
- "用户希望..."
- "用户要求不要..."

They should not use overconfident claims like "用户一定..." or "用户患有...".

## Optional LLM Consolidation

Later, `MEMORY_CONSOLIDATION_MODE=llm` can use the configured LLM to distill the recent window. It should be asynchronous from the user's perspective: chat returns normally, and consolidation failures are warnings.

The LLM prompt should ask for structured JSON candidates with:

- `content`
- `category`
- `kind`
- `confidence`
- `evidence`
- `sensitivity`

The parser must discard malformed candidates, unsupported categories, low-confidence items, and candidates that look like diagnosis or unsupported inference.

The LLM path must still write through the same `MemoryProvider.remember()` method so duplicate, conflict, recency, and supersession behavior remains centralized.

## Retrieval Improvements

Current retrieval uses the current user message as the query. For emotional companionship, the query should include a small amount of emotional context:

```text
<current user message>
Current emotion: anxious
Recent emotions: anxious, sad
```

This can improve recall for short messages like "又来了" or "还是那个问题" when the current emotion and recent history point to a known pattern.

The search should remain bounded and local. If no relevant memories match, the memory context should stay empty.

## Prompt Injection

Keep the existing `Relevant Long-term Memory:` format for the first implementation to minimize risk.

Within that stable format, sort memories before formatting:

1. `boundary`
2. `preference`
3. `goal`
4. `profile`

If metadata is later exposed to formatting, emotional-support memories can be grouped more clearly, but that is not required for the first implementation.

The assistant should treat long-term memory as helpful context, not absolute truth, except for explicit boundaries.

## Data Model

The existing `memories.metadata_json` column is sufficient for the first version.

Add a small consolidation state table:

```text
memory_consolidation_state
  id text primary key
  last_turn_count integer not null
  last_message_id text
  updated_at text not null
```

For the current single-user app, the row id can be `default`. If the app later supports multiple users, this can become a user or session id.

This state prevents repeated consolidation of the same recent window and gives tests a stable checkpoint to assert against.

## Chat Service Flow

`ChatService.generate_reply()` and `ChatService.stream_reply()` should continue to prioritize response generation.

The updated flow:

1. Append user message.
2. Refresh memory context using an enhanced retrieval query.
3. Run emotion analysis if due.
4. Apply turn safety.
5. Generate or stream the assistant reply.
6. Append AI message.
7. Extract immediate memory candidates from the current turn.
8. If consolidation is due, run a bounded consolidation pass.

If consolidation fails, print a warning and continue. It must not change the already-returned response.

## Module Boundaries

Add a focused module:

```text
chatbot/memory_consolidation.py
```

Responsibilities:

- decide whether consolidation is due
- build a recent conversation window
- produce rule-based consolidation candidates
- expose an interface that can later support LLM-based consolidation

Keep SQLite-specific state in `chatbot.local_memory` or a small helper owned by the local provider. `ChatService` should not contain SQL or consolidation heuristics.

Suggested public functions or classes:

```python
@dataclass(frozen=True)
class MemoryConsolidationConfig:
    enabled: bool
    interval: int
    window: int
    mode: str

def load_memory_consolidation_config(...) -> MemoryConsolidationConfig:
    ...

def build_memory_search_query(message: str, emotion_state, recent_emotions: list[str]) -> str:
    ...

def extract_consolidated_memory_candidates(records: list[dict]) -> list[MemoryCandidate]:
    ...
```

## Testing

Add tests that do not require live LLM calls:

- consolidation config defaults and invalid-value fallback
- consolidation disabled when memory is disabled
- search query includes current message, current emotion, and recent emotions
- rule consolidation extracts an explicit support preference from a recent window
- rule consolidation ignores a single transient feeling
- repeated stressor mentions become a cautious `profile` memory
- boundaries from the consolidation window become `boundary` candidates
- chat still succeeds when consolidation raises an exception
- consolidation is triggered only at the configured interval
- existing memory provider conflict behavior still applies to consolidated candidates

## Success Criteria

This design is successful when:

- the chatbot remembers durable emotional-support preferences without storing every passing mood
- current emotion and recent emotion trajectory improve long-term memory retrieval
- consolidation never blocks or breaks chat
- memory remains local-first and SQLite-backed by default
- Mem0 remains unnecessary for the first implementation
- tests can verify the behavior without real LLM calls

