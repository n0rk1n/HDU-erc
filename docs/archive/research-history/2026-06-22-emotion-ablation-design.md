# Emotion Recognition Ablation Design

## Goal

Prepare a reproducible first-stage ablation experiment for the main chatbot application's emotion-recognition pipeline.

The first stage focuses on quantitative emotion-recognition quality. It should produce comparable accuracy and macro F1 results for several controlled variants of the current prompt and context strategy. Chat response quality evaluation is intentionally deferred to a later stage.

## Scope

This design covers the main application under `chatbot/`, not the standalone EICL paper reproduction code under `EICL/`.

The experiment should answer how much these pieces contribute to emotion recognition:

- Dynamic few-shot example retrieval.
- Previous and likely emotion history.
- Recent dialogue context length.
- The full current prompt strategy compared with a zero-shot baseline.

## Experiment Matrix

All runs should use the same test data, model provider, model name, temperature, and label file. Each run writes its own `emotion_analysis.json` style output file.

| Run | Dialogue Context | Few-Shot Examples | Previous/Likely Emotion | Purpose |
| --- | --- | --- | --- | --- |
| `full` | Default `EMOTION_INTERVAL` window | Dynamic selection | Enabled | Current full strategy |
| `no_dynamic_examples` | Default window | Static or empty examples | Enabled | Measure dynamic EICL retrieval contribution |
| `no_emotion_history` | Default window | Dynamic selection | Disabled | Measure emotion-history prior contribution |
| `short_context` | One-turn window | Dynamic selection | Enabled | Measure recent-context length contribution |
| `zero_shot` | Default window | None | Disabled | Weak baseline |

## Data Files

Use two fixed JSONL files for the first-stage dataset:

- `data/examples/ablation_dialogues.jsonl`
- `data/examples/ablation_labels.jsonl`

`ablation_dialogues.jsonl` contains the inputs needed to run emotion recognition:

```json
{"id":"case-001","turn_count":1,"history":[],"current_input":"I am so nervous about tomorrow's presentation.","notes":"single-turn anxiety"}
{"id":"case-002","turn_count":3,"history":[{"role":"human","content":"I studied all week."},{"role":"ai","content":"That sounds like real effort."}],"current_input":"I still feel like I might fail.","notes":"anxiety after effort"}
```

`ablation_labels.jsonl` contains manual labels:

```json
{"id":"case-001","expected":"anxious"}
{"id":"case-002","expected":"anxious"}
```

The fixed test set should be used first to validate the pipeline. A sampled and manually labeled subset of real runtime records can be added later as an external validation set.

## Output Files

Each run writes a separate analysis file under `data/records/ablation/`:

- `data/records/ablation/full.json`
- `data/records/ablation/no_dynamic_examples.json`
- `data/records/ablation/no_emotion_history.json`
- `data/records/ablation/short_context.json`
- `data/records/ablation/zero_shot.json`

Records should stay compatible with the current evaluation scripts and include `case_id` and `run` for stable matching and traceability:

```json
{
  "turn_count": 3,
  "emotion_interval": 5,
  "input": "Dialogue context: I studied all week.</s>That sounds like real effort.</s>I still feel like I might fail.",
  "output": "Emotion: anxious",
  "emotion": "anxious",
  "state": {"primary_emotion": "anxious"},
  "success": true,
  "error": "",
  "case_id": "case-002",
  "run": "full"
}
```

## Running The Comparison

The existing multi-run evaluator should remain the reporting entry point:

```bash
python scripts/evaluate_emotion_ablation.py \
  --labels-file data/examples/ablation_labels.jsonl \
  --run full=data/records/ablation/full.json \
  --run no_dynamic_examples=data/records/ablation/no_dynamic_examples.json \
  --run no_emotion_history=data/records/ablation/no_emotion_history.json \
  --run short_context=data/records/ablation/short_context.json \
  --run zero_shot=data/records/ablation/zero_shot.json \
  --markdown-file data/records/ablation/summary.md \
  --csv-file data/records/ablation/metrics.csv
```

The report table should use `summary.md`; `metrics.csv` and mismatch examples should be retained for inspection.

## Proposed Module Boundaries

Add `scripts/run_emotion_ablation.py` as a small first-stage runner. It should:

- Read `ablation_dialogues.jsonl`.
- Load model configuration with the existing config and LLM adapter path.
- Build emotion prompts for each run configuration.
- Invoke the emotion LLM.
- Parse output with existing emotion-state parsing.
- Write one output file per run.

It should not start FastAPI, write the default runtime `data/records/emotion_analysis.json`, or change user chat history.

Extend the emotion prompt path with controlled options while preserving default Web behavior:

- `example_mode`: `dynamic`, `static`, or `none`.
- `include_emotion_history`: `true` or `false`.
- `max_turns`: default interval or one-turn context.

Enhance `scripts/evaluate_emotion_analysis.py` matching so annotations can match records by `id` or `case_id` before falling back to existing matching rules. The preferred priority is:

1. `id` or `case_id`.
2. `index`.
3. `turn_count`.
4. `timestamp`.
5. Successful-record order.

## Error Handling

One failed case should not abort a whole run. The runner should write a failed record:

```json
{
  "case_id": "case-001",
  "emotion": "",
  "success": false,
  "error": "LLM request timed out"
}
```

The evaluator should treat this as a missing prediction and count it as an error for that labeled sample.

Output directories should be created automatically. Invalid JSONL, missing labels, and unsupported run names should fail clearly before model calls begin.

## Tests

The first-stage implementation should add focused tests for:

- `build_emotion_prompt` or its prompt-building helper omits examples when `example_mode=none`.
- Emotion history is not injected when `include_emotion_history=false`.
- Evaluation matches labels to records by `case_id`.
- The ablation runner can use a fake LLM to generate multiple run files.
- CLI help works.
- Missing output directories are created automatically.

## Non-Goals

- Do not evaluate chat response quality in the first stage.
- Do not add Web UI.
- Do not change the default chat behavior.
- Do not require downloading or reproducing the full EICL paper experiment stack.
- Do not overwrite real runtime records.

## Acceptance Criteria

- A fixed JSONL test set and label file can drive all first-stage runs.
- Each ablation run writes an isolated analysis file.
- Multi-run evaluation produces Markdown and CSV summaries with accuracy and macro F1.
- Existing chatbot behavior remains unchanged unless the new runner is invoked.
- Unit tests cover the prompt switches, case-id matching, and runner output behavior.
