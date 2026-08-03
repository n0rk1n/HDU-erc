# Local Memory v2 Design

## Goal

Improve the chatbot's local long-term memory so it more accurately preserves durable user preferences and constraints, and retrieves more relevant memories during chat.

This design keeps the current local-first architecture: SQLite storage, no third-party hosted memory service, and the existing `MemoryProvider` boundary used by `ChatService`.

## Current State

The current memory flow is intentionally conservative:

1. `ChatService` receives a user message.
2. Before generating a reply, it calls `memory_provider.search(message, limit=...)`.
3. Matching memories are formatted by `format_memory_context()` and injected into the chat prompt.
4. After the assistant reply completes, `memory_extractor.extract_memory_candidates()` extracts durable memory candidates from the user message.
5. `memory_provider.remember(candidates)` writes candidates to SQLite.

This design is stable and easy to test, but it has three important limits:

- Search uses lexical token overlap, so semantic matches are often missed.
- Duplicate detection only catches exact normalized text matches.
- Conflicting preferences can coexist and both be injected into the prompt.

## Non-Goals

This version will not add embedding search, a hosted memory platform, or LLM-based memory extraction.

It will not add a memory management UI. User-visible review, edit, and delete controls can be designed separately after the memory core is more reliable.

## Architecture

The existing module boundaries should stay intact:

- `chatbot.memory_extractor`: extracts conservative memory candidates.
- `chatbot.local_memory`: owns SQLite schema, migrations, duplicate detection, conflict handling, and retrieval ranking.
- `chatbot.memory`: owns shared data types, provider protocol, config, and prompt formatting.
- `chatbot.chat_service`: continues to call only `search()` before response generation and `remember()` after response completion.

Most changes belong in `chatbot.local_memory`. `ChatService` should not learn about scoring rules, conflict rules, or SQLite migration details.

## Data Model

Add three columns to the `memories` table:

```text
status text not null default 'active'
supersedes_id text
metadata_json text not null default '{}'
```

`status` controls whether a memory participates in retrieval. Local Memory v2 supports:

- `active`: eligible for search and prompt injection.
- `superseded`: retained for auditability but excluded from search.

`supersedes_id` records the replaced memory when a new memory supersedes an older one.

`metadata_json` stores lightweight internal metadata such as conflict reason, normalized topic, or matched tokens. It is intentionally a JSON string to avoid frequent schema migrations while the memory rules are still evolving.

The public `Memory` dataclass does not need to expose these fields in v2 because `ChatService` and prompt formatting only need active memories. Provider internals can read and write the extra columns.

## Migration

`SQLiteLocalMemoryProvider._init_schema()` should become migration-aware:

1. Create the table if it does not exist.
2. Inspect existing columns with SQLite metadata.
3. Add missing v2 columns with `alter table`.
4. Treat all pre-existing rows as `active`.

The migration must be automatic on provider initialization and must preserve existing records.

## Write Path

Memory extraction stays conservative and rule-based. The improvement happens inside `remember()` before insert or update.

### Normalization

Normalize candidate content for comparison more strongly than the stored content:

- Collapse whitespace.
- Ignore trailing sentence punctuation for comparison.
- Compare English text case-insensitively.
- Preserve original stored text.
- Keep stable Chinese template prefixes such as `用户希望`, `用户喜欢`, and `用户要求不要` meaningful during comparison.

### Duplicate And Similar Memory Handling

When a candidate arrives, search existing active memories in the same category first.

Update an existing row instead of inserting a new row when:

- The normalized content is identical.
- The candidate and existing memory have strong token overlap.
- The candidate and existing memory match the same known preference topic, such as concise replies, detailed replies, Chinese replies, English replies, formal tone, casual tone, or hosted storage boundaries.

When updating, preserve the original `id` and `created_at`, update `content`, `category`, `source`, `confidence`, and `updated_at`, and keep the maximum confidence.

### Conflict Handling

Support a small, explicit conflict rule set:

- concise replies vs detailed replies
- Chinese replies vs English replies
- formal tone vs casual tone
- hosted or third-party memory storage preference vs hosted or third-party memory storage boundary

If a new active memory conflicts with an existing active memory, mark the older memory as `superseded` and insert or update the new memory as `active`.

`boundary` memories are stronger than ordinary `preference` memories. A boundary must not be superseded by a normal preference unless the new candidate is also an explicit boundary or explicit reversal. Local Memory v2 only needs the conservative behavior: keep the boundary active when a weaker preference conflicts with it.

The guiding principle is to avoid injecting contradictory long-term preferences into the prompt.

## Search Path

Search should continue to require lexical relevance, but ranking should be more expressive and explainable.

Only `active` memories participate in search.

For each candidate memory with a basic lexical match, compute a score from:

```text
score =
  lexical_match_score
  + category_weight
  + confidence_weight
  + recency_weight
  + usage_weight
```

`lexical_match_score` should extend the existing token overlap approach. Chinese bigrams and exact phrase matches should carry more weight than incidental single-token overlap.

`category_weight` should prioritize behavioral constraints:

- `boundary`: strongest
- `preference`: next
- `goal` and `profile`: lower by default

`confidence_weight` should give high-confidence memories a modest boost.

`recency_weight` should help newer active preferences beat older ones.

`usage_weight` should be small so early memories do not become permanently dominant just because they were retrieved often.

The provider should continue to mark returned memories as used by updating `last_used_at` and incrementing `use_count`.

## Prompt Injection

The prompt format should remain unchanged:

```text
Relevant Long-term Memory:
- 用户希望回答简洁。
- 用户要求不要使用第三方托管记忆服务。
```

Keeping the prompt format stable reduces risk and keeps this design focused on memory quality rather than prompt redesign.

## Error Handling

Memory search and write failures should continue to be non-fatal. The chatbot should keep responding if memory operations fail.

SQLite errors should be caught inside the provider and logged as warnings, matching the current behavior.

Migration failures should also fail closed: memory operations can return no results or no stored rows, but chat should continue.

## Testing

Add focused tests around the provider and extractor boundary:

- Existing exact duplicate behavior still updates one row.
- Old SQLite schemas are migrated automatically.
- Similar preferences such as concise-answer variants merge instead of duplicating.
- Conflicting preferences such as detailed replies followed by concise replies return only the newer active memory.
- `boundary` memories are not superseded by weaker ordinary preferences.
- `superseded` memories are excluded from search.
- Search ranks stronger matches, higher-priority categories, and newer active memories ahead of weaker matches.
- Search still returns an empty list when there is no lexical relevance.

Existing `ChatService` tests should continue to pass without major changes because the public provider contract remains the same.

## Success Criteria

Local Memory v2 is successful when:

- The chatbot no longer injects obvious contradictory preferences after a user changes their mind.
- Similar durable preferences do not accumulate as noisy duplicates.
- Relevant preferences and boundaries are more likely to appear near the top of retrieved memory context.
- The system remains local-first, low-dependency, and non-fatal when memory operations fail.
- The implementation is covered by focused unit tests without requiring live LLM calls.
