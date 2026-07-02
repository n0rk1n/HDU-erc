# Bilingual Parallel Policy

`core_parallel` cases must preserve emotion evidence across Chinese and English while remaining natural in each language.

Requirements:

- `pair_id` links the Chinese and English instances in the same parallel pair.
- Primary labels must match across the pair.
- Intensity difference should be no more than `0.15`.
- Context dependency should match or be adjacent on the `none`, `low`, `medium`, `high` scale.
- Evidence spans should correspond semantically, even when wording is not literal.
- Each side must sound like natural language for that locale, not a mechanical translation.

If direct translation changes emotion strength, cultural implication, or user intent, rewrite both sides before release. Reject or revise pairs with unresolved `parallel_mismatch` flags.
