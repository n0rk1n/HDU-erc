# Emotion-Aware Chatbot v2 Design

## Goal

Upgrade the current emotion-recognition chatbot into a research-oriented and demo-friendly system for student-to-advisor reporting.

The v2 system should connect the existing Web chatbot with the EICL research direction, improve emotion recognition quality, expose interpretable emotion state, show emotion changes over time, collect feedback, add basic safety handling, and provide ablation evaluation evidence.

The central contribution is:

> Apply EICL-style dynamic emotion example selection to an online chatbot, then use structured emotion state to guide responses, timeline display, feedback collection, safety handling, and evaluation.

## Current State

The repository already has a solid application baseline:

- FastAPI Web app with SSE streaming replies.
- Persistent local chat history.
- Periodic emotion recognition through an LLM.
- Fixed emotion label set.
- Few-shot emotion prompt examples.
- Recent successful emotion labels used as likely candidates.
- Local SQLite long-term memory.
- AI message feedback and regeneration controls.
- Emotion evaluation script with accuracy and macro F1.
- Original EICL paper code and experiment assets under `EICL/`.

The main limits are:

- Emotion recognition returns one parsed label, so it is hard to explain why a label was selected.
- Few-shot examples are static rather than dynamically selected from the current dialogue.
- Emotion history is stored, but not modeled as a user-facing timeline or transition signal.
- Feedback is mostly attached to AI messages, not emotion correctness.
- Evaluation measures stored predictions, but does not compare ablation variants.
- Safety behavior is not explicit for strongly negative or crisis-like user messages.

## Non-Goals

This version will not train a new neural model.

It will not require a hosted vector database or third-party memory platform.

It will not rebuild the full EICL experiment pipeline inside the chatbot request path.

It will not make medical or psychological diagnoses. Safety handling is limited to supportive response guidance and escalation suggestions for high-risk text.

## Architecture

Keep `ChatService` as the orchestration layer, but introduce a richer emotion subsystem.

Core boundaries:

- `chatbot.chat_service`: coordinates memory search, emotion analysis, safety policy, reply generation, history, memory writes, and feedback signals.
- `chatbot.emotion`: public emotion-analysis entry point and persistence compatibility.
- `chatbot.emotion_prompt`: structured prompt construction and output instructions.
- `chatbot.emotion_examples`: labeled example library.
- `chatbot.emotion_retrieval`: dynamic EICL-style example selection.
- `chatbot.emotion_state`: structured result models, timeline helpers, transition summaries, and serialization.
- `chatbot.safety`: lightweight safety policy based on user text and emotion state.
- `chatbot.web`: API routes for session, emotion timeline, feedback, and static UI.
- `scripts/ablation/evaluate_emotion_analysis.py`: remains the single-run evaluator.
- `scripts/ablation/evaluate_emotion_ablation.py`: compares multiple recognition variants.

The existing local-first storage model should stay intact. JSON files and SQLite remain acceptable for v2.

## Structured Emotion State

Replace the internal single-label emotion result with a structured state while preserving backward-compatible access to the primary emotion.

The structured state should contain:

```json
{
  "primary_emotion": "anxious",
  "confidence": 0.78,
  "secondary_emotions": ["apprehensive", "sad"],
  "evidence": "The user expresses uncertainty and fear of failure.",
  "reply_strategy": "Use a calm and supportive tone before giving concrete suggestions.",
  "trajectory_note": "The user shifted from hopeful to anxious.",
  "safety_level": "normal"
}
```

Validation rules:

- `primary_emotion` must be in the existing label set.
- `secondary_emotions` must contain only known labels after filtering.
- `confidence` is clamped to `0.0` through `1.0`.
- `evidence`, `reply_strategy`, and `trajectory_note` are short strings.
- Invalid or unparsable model output records a failed analysis and does not block chat.

The persisted `emotion_analysis.json` record should include both legacy fields and the new structured payload:

- `emotion`: same as `primary_emotion`.
- `success`: existing success flag.
- `output`: raw model output.
- `state`: structured normalized payload.
- `examples`: selected dynamic examples or their ids.
- `safety`: safety policy output.

## Dynamic EICL Example Selection

Add a lightweight retrieval layer that selects few-shot examples dynamically for each emotion-analysis call.

Input:

- Recent dialogue contents.
- Current user input.
- Previous primary emotion.
- Recent successful emotions.
- Static labeled examples from `emotion_examples`.

Output:

- Three to five selected examples.
- Simple score and reason for each selected example.

The first implementation should use explainable local retrieval:

- Exact emotion-prior boost for recent emotions.
- Token overlap between current dialogue and example text.
- Chinese or English phrase overlap where available.
- Diversity rule so all examples do not collapse to the same emotion.

Embedding search can be a later extension. Keeping v2 lexical and local makes the implementation easier to test and explain in a student report.

The prompt should present selected examples as dynamic EICL examples, not as a fixed hand-picked prompt block.

## Emotion Timeline

Maintain a timeline of successful structured emotion states.

The timeline should support:

- Current primary emotion.
- Confidence.
- Secondary emotions.
- Evidence summary.
- Reply strategy.
- Safety level.
- Recent transitions, such as `hopeful -> anxious`.

Back-end access should use `GET /api/emotion/timeline` for the full timeline. The existing `/api/session` response can keep returning only the latest emotion summary so page initialization stays lightweight.

The front end should show:

- Current emotion badge.
- Confidence value.
- Recent timeline list.
- Compact transition text.

This display is important for advisor demos because it makes emotion modeling visible instead of hidden inside logs.

## Chat Prompt Integration

The chat prompt should receive richer emotion context, but the integration should stay concise.

The generated context should include:

- Primary emotion.
- Confidence.
- Secondary emotions when present.
- Short evidence.
- Reply strategy.
- Safety guidance when safety level is not normal.

Example:

```text
Current Emotion:
- primary: anxious
- confidence: 0.78
- secondary: apprehensive, sad
- evidence: The user expresses uncertainty and fear of failure.
- reply strategy: Use a calm and supportive tone before giving concrete suggestions.
```

`ChatService` should continue passing an `emotion_context` string to the chain. Prompt-format details belong in `llm.py` or a small formatter helper, not in the service orchestration code.

## Safety Policy

Add `chatbot.safety` with a conservative local policy.

Safety levels:

- `normal`: ordinary chat.
- `supportive`: high-intensity negative emotion or distress, but no direct crisis language.
- `crisis`: explicit self-harm, suicide, or immediate danger language.

The policy should combine:

- Keyword and phrase rules on the current user message.
- Primary and secondary emotions.
- Confidence.
- Recent trajectory if available.

For `supportive`, the reply strategy should encourage warmth, validation, and practical next steps.

For `crisis`, the assistant should avoid diagnosis, avoid minimizing the user, encourage contacting trusted people or local emergency/professional support, and keep the response grounded and immediate.

Safety checks must not block the app unless a future requirement explicitly asks for blocking behavior.

## Feedback Loop

Extend feedback beyond AI message quality.

Add emotion-correctness feedback for recent AI turns or the current session state:

- `accurate`
- `too_positive`
- `too_negative`
- `wrong_emotion`

Persist this locally with:

- Related message id if available.
- Turn count.
- Predicted emotion.
- Optional corrected emotion label.
- Timestamp.

This feedback will not automatically retrain or rewrite model behavior in v2. It is used for:

- Manual evaluation labels.
- Regeneration context.
- Future prompt or retrieval improvements.

The existing message feedback and regeneration code should remain intact. Emotion feedback is a new parallel signal.

## Ablation Evaluation

Add `scripts/ablation/evaluate_emotion_ablation.py` to compare multiple recognition modes.

Recommended modes:

- `zero-shot`: no examples.
- `static-few-shot`: current fixed examples.
- `static-few-shot-candidates`: fixed examples plus recent emotion candidates.
- `dynamic-eicl`: dynamic example retrieval.
- `dynamic-eicl-trajectory`: dynamic examples plus previous emotion and trajectory prior.

Metrics:

- Samples.
- Correct count.
- Accuracy.
- Macro F1.
- Per-label precision, recall, and F1.
- Error examples.

Outputs:

- Console summary.
- Markdown table for reports.
- CSV for slides or spreadsheets.

The evaluator should be deterministic and testable without live LLM calls. It should compare saved prediction files or saved analysis records from each mode.

## Web UI

The UI should stay a practical chat application, not become a research dashboard.

Add compact controls and displays:

- Current emotion badge near the chat header.
- Confidence text or small meter.
- Recent emotion timeline panel.
- Emotion correctness feedback near AI feedback controls.
- Safety state indicator only when supportive or crisis guidance is active.

Avoid heavy explanatory copy in the app. The UI should reveal state through controls and compact labels.

## Data Flow

```text
User message
  -> append user history
  -> memory search
  -> dynamic EICL example selection
  -> structured emotion analysis when due or triggered
  -> safety policy check
  -> timeline update
  -> chat prompt context assembly
  -> streaming AI reply
  -> append AI history
  -> conservative memory extraction
  -> message feedback and emotion feedback collection
```

Emotion analysis remains non-fatal. If it fails, the chat response should continue with the previous successful emotion state or no emotion context.

## Implementation Phases

### Phase 1: Structured Emotion State

- Add `EmotionState`.
- Parse structured model output.
- Persist `state` while keeping legacy `emotion`.
- Format richer emotion context for chat.
- Test parsing, fallback, persistence, and context formatting.

### Phase 2: Dynamic EICL Retrieval

- Add `emotion_retrieval`.
- Select dynamic examples from the local example library.
- Add prompt support for dynamic examples.
- Test ranking, diversity, candidate boosts, and fallback behavior.

### Phase 3: Timeline and UI

- Add timeline helpers and API response.
- Show current emotion and recent transitions in the Web UI.
- Test API shape and static JavaScript behavior where practical.

### Phase 4: Safety Policy

- Add safety classifier rules.
- Inject safety guidance into reply strategy.
- Test normal, supportive, and crisis-like cases.

### Phase 5: Feedback Loop

- Add emotion feedback persistence.
- Add API route and UI controls.
- Connect emotion feedback to evaluation-label export or saved records.
- Test feedback validation and persistence.

### Phase 6: Ablation Evaluation

- Add ablation evaluator.
- Define saved prediction file format.
- Produce Markdown and CSV summaries.
- Test metric calculations and mismatched record handling.

## Testing Strategy

Use unit tests for most behavior:

- Structured output parser accepts valid JSON and structured text fallback.
- Invalid labels fail closed.
- Confidence is normalized.
- Dynamic retrieval returns relevant, diverse examples.
- Safety policy detects normal, supportive, and crisis cases.
- Timeline serializes recent states correctly.
- Emotion context remains concise and includes reply strategy.
- Feedback APIs reject invalid values.
- Ablation evaluator computes accuracy, macro F1, per-label metrics, and errors.

Use Web tests for:

- Timeline endpoint returns the recent emotion timeline.
- Emotion feedback route persists valid feedback.
- Existing chat streaming behavior still works.

No tests should require live LLM calls.

## Success Criteria

The v2 work is successful when:

- The chatbot produces structured, explainable emotion states.
- Emotion recognition can use dynamic EICL-style examples.
- The Web UI visibly shows current emotion and recent emotion changes.
- Chat replies can adapt to emotion state and safety guidance.
- Users can provide emotion-correctness feedback.
- Ablation evaluation can compare baseline, static few-shot, and dynamic EICL modes.
- Existing local-first behavior, history persistence, message feedback, regeneration, and memory features continue to work.
- The feature set can be presented as both a technical contribution and a working demo for an advisor report.
