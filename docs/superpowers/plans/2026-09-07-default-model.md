# Default model configuration implementation plan

Goal: Keep LLM_* as shared defaults; CHAT_LLM_*, EMOTION_LLM_* and EMOTION_GATE_LLM_* override these independently, field by field. Missing and whitespace-only overrides inherit defaults.

Architecture: A shared model settings resolver in chatbot/core/config.py validates connection settings. Existing AppConfig, emotion settings and gate settings consume it without changing their public loader signatures. Emotion token budgets remain explicit; gate tokenizer compatibility checks remain enforced.

Scope: Update the three configuration loaders, regression tests, .env.example and README.md. Never copy local credentials into the task or commit .env.

- [x] Add tests/core/test_model_defaults.py covering unset/blank overrides, independent roles, partial overrides and invalid parameters. Run it and verify current behavior fails.
- [x] Implement resolve_model_settings(prefix) returning validated ModelSettings with api_key, model, base_url, temperature and timeout_seconds. Route all three loaders through it.
- [x] Adjust gate config fixtures to supply the shared default connection explicitly. Update environment template and README inheritance guidance.
- [x] Run configuration tests and the full offline suite, inspect diff and status, and commit task changes. Keep task branch for user merge.

Execution: Inline in this task, per repository policy; no subagents.

Verification: 306 passed, 1 live test skipped. Bundled Node was added to PATH for the existing JavaScript harness. One existing Starlette deprecation warning. git diff --check passed.
