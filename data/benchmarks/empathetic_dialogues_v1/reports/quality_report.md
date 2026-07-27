# Quality Report

- Official source archive SHA-256 verified before conversion.
- Parsed test conversations: 2,542.
- Project-schema validation errors: 0.
- Label coverage: 32 of 32 project labels.
- Balanced seed: 64 records, exactly 2 per label.
- Placeholder cleanup: `_comma_` restored to commas.
- Leakage control: source situation prompt retained only as provenance and not
  exported into the model dialogue input.
- Context distribution: `none=4`, `low=2`, `medium=1984`, `high=552`.
- Original-method Codex pilot: 10 records × 5 configurations, 50/50 valid
  predictions and 0 invocation failures; detailed under `codex_pilot10/`.

No claim is made that the source conversation-level labels were independently
re-annotated or adjudicated at utterance level.
