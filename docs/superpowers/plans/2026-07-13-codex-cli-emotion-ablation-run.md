# Codex CLI Emotion Ablation Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable Codex CLI adapter, run the five existing emotion-recognition ablations on a 10-case pilot and the 64-case v2 seed set, and generate a verified Chinese report.

**Architecture:** A focused runner reuses `scripts.run_emotion_ablation.RUN_CONFIGS` and the existing prompt builder, but replaces the OpenAI-compatible client with one isolated `codex exec` subprocess per case. A separate report module reads the five JSON outputs and seed metadata, computes global and sliced metrics with the existing evaluator, and renders CSV, Markdown summary, and a Chinese narrative report.

**Tech Stack:** Python 3.12, standard library `argparse/json/subprocess/csv`, existing emotion prompt and evaluation modules, Codex CLI 0.142.4, pytest.

## Global Constraints

- Run the first 10 seed records as a 50-task pilot before the 64-record, 320-task formal run.
- Use an independent `codex exec --ephemeral --sandbox read-only` session for every case.
- Use the same Codex model/configuration for every ablation group.
- Retry a failed task at most once and resume by skipping successful case/run pairs.
- Count failed predictions as errors and report them separately.
- Do not modify chatbot runtime behavior or overwrite existing runtime emotion records.
- Do not run the 500-record formal release in this iteration.

---

## File Structure

- Create `data/config/codex_emotion_result.schema.json`: strict structured-output schema for one emotion label.
- Create `scripts/run_codex_cli_emotion_ablation.py`: command construction, Codex invocation, parsing, retry/resume, and per-run JSON output.
- Create `tests/test_run_codex_cli_emotion_ablation.py`: unit tests with a fake subprocess runner plus CLI-help coverage.
- Create `scripts/report_codex_cli_emotion_ablation.py`: global/sliced metrics and Chinese Markdown/CSV rendering.
- Create `tests/test_report_codex_cli_emotion_ablation.py`: deterministic report and metric tests.
- Generate `data/records/codex_cli_ablation/pilot/*.json`: 10-case pilot outputs.
- Generate `data/records/codex_cli_ablation/seed64/*`: 64-case outputs, metrics, summary, and Chinese report.

### Task 1: Codex CLI Structured Invocation

**Files:**
- Create: `data/config/codex_emotion_result.schema.json`
- Create: `scripts/run_codex_cli_emotion_ablation.py`
- Test: `tests/test_run_codex_cli_emotion_ablation.py`

**Interfaces:**
- Consumes: `RUN_CONFIGS`, `load_dialogues`, `_history_records`, `_previous_emotion`, `_likely_emotions`, and `_turn_count` from `scripts.run_emotion_ablation`; `build_emotion_prompt` and `EMOTION_LABEL_SET` from `chatbot`.
- Produces: `CodexResult`, `build_command()`, `parse_result()`, and `invoke_codex()` for Task 2.

- [ ] **Step 1: Add failing command and parser tests**

```python
def test_build_command_uses_isolated_structured_exec(tmp_path):
    command = runner.build_command(tmp_path / "schema.json", model="gpt-test")
    assert command == [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--output-schema", str(tmp_path / "schema.json"),
        "--model", "gpt-test", "-",
    ]


def test_parse_result_accepts_supported_label():
    result = runner.parse_result('{"emotion":"anxious"}')
    assert result.emotion == "anxious"
    assert result.success is True


def test_parse_result_rejects_unknown_label():
    result = runner.parse_result('{"emotion":"relieved"}')
    assert result.success is False
    assert "Unsupported emotion" in result.error
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `./.venv312/bin/python -m pytest tests/test_run_codex_cli_emotion_ablation.py -q`

Expected: FAIL during import because `scripts.run_codex_cli_emotion_ablation` does not exist.

- [ ] **Step 3: Add the strict output schema**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "emotion": {
      "type": "string",
      "enum": ["surprised", "excited", "annoyed", "proud", "angry", "sad", "grateful", "lonely", "impressed", "afraid", "disgusted", "confident", "terrified", "hopeful", "anxious", "disappointed", "joyful", "prepared", "guilty", "furious", "nostalgic", "jealous", "anticipating", "embarrassed", "content", "devastated", "sentimental", "caring", "trusting", "ashamed", "apprehensive", "faithful"]
    }
  },
  "required": ["emotion"],
  "additionalProperties": false
}
```

- [ ] **Step 4: Implement command construction and strict parsing**

```python
@dataclass(frozen=True)
class CodexResult:
    emotion: str
    output: str
    success: bool
    error: str = ""


def build_command(schema_file: Path, *, model: str | None) -> list[str]:
    command = [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--output-schema", str(schema_file),
    ]
    if model:
        command.extend(["--model", model])
    command.append("-")
    return command


def parse_result(output: str) -> CodexResult:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return CodexResult("", output, False, f"Invalid JSON: {exc}")
    emotion = payload.get("emotion") if isinstance(payload, dict) else None
    if emotion not in EMOTION_LABEL_SET:
        return CodexResult("", output, False, f"Unsupported emotion: {emotion!r}")
    return CodexResult(emotion, output, True)
```

- [ ] **Step 5: Implement one retryable Codex invocation**

```python
def invoke_codex(
    prompt: str,
    *,
    schema_file: Path,
    model: str | None,
    timeout: int,
    run_subprocess=subprocess.run,
) -> CodexResult:
    instruction = (
        "Classify the emotion using only the supplied prompt. "
        "Return exactly the JSON object required by the output schema.\n\n" + prompt
    )
    try:
        completed = run_subprocess(
            build_command(schema_file, model=model),
            input=instruction,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return CodexResult("", "", False, str(exc))
    if completed.returncode != 0:
        return CodexResult("", completed.stdout, False, completed.stderr.strip())
    return parse_result(completed.stdout.strip())
```

- [ ] **Step 6: Run the focused tests**

Run: `./.venv312/bin/python -m pytest tests/test_run_codex_cli_emotion_ablation.py -q`

Expected: PASS for command, parser, subprocess success, timeout, and CLI-help tests.

- [ ] **Step 7: Commit Task 1**

```bash
git add data/config/codex_emotion_result.schema.json scripts/run_codex_cli_emotion_ablation.py tests/test_run_codex_cli_emotion_ablation.py
git commit -m "feat: add codex cli emotion adapter"
```

### Task 2: Resumable Five-Group Runner

**Files:**
- Modify: `scripts/run_codex_cli_emotion_ablation.py`
- Modify: `tests/test_run_codex_cli_emotion_ablation.py`

**Interfaces:**
- Consumes: `invoke_codex(prompt, schema_file, model, timeout)` from Task 1 and existing `RUN_CONFIGS`.
- Produces: `run_ablation()` and CLI options `--dialogues-file`, `--output-dir`, `--limit`, `--run`, `--model`, `--timeout`, and `--retries`.

- [ ] **Step 1: Add failing resume, retry, and prompt-variant tests**

```python
def test_run_ablation_skips_existing_success_and_retries_failure(tmp_path):
    output_file = tmp_path / "full.json"
    output_file.write_text(json.dumps([{
        "case_id": "case-001", "run": "full", "emotion": "anxious", "success": True
    }]), encoding="utf-8")
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        return runner.CodexResult("grateful", '{"emotion":"grateful"}', True)

    records = runner.run_ablation(
        runner.RUN_CONFIGS["full"], CASES, output_file,
        schema_file=tmp_path / "schema.json", model=None, timeout=60,
        retries=1, invoke=fake_invoke,
    )
    assert [record["case_id"] for record in records] == ["case-001", "case-002"]
    assert len(calls) == 1
```

Add assertions that `no_emotion_history` omits likely emotions, `short_context` only includes the final turn, and `zero_shot` omits examples.

- [ ] **Step 2: Run the new tests and confirm failure**

Run: `./.venv312/bin/python -m pytest tests/test_run_codex_cli_emotion_ablation.py -q`

Expected: FAIL because `run_ablation` and new CLI options are absent.

- [ ] **Step 3: Implement resumable execution and atomic snapshots**

```python
def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_ablation(config, cases, output_file, *, schema_file, model, timeout, retries, invoke=invoke_codex):
    existing = _load_existing(output_file)
    by_case = {record["case_id"]: record for record in existing if record.get("success") is True}
    for index, case in enumerate(cases, start=1):
        case_id = case["id"].strip()
        if case_id in by_case:
            continue
        history = _history_records(case)
        prompt = build_emotion_prompt(
            history,
            case["current_input"],
            previous_emotion=_previous_emotion(history),
            likely_emotions=_likely_emotions(history),
            max_turns=config.max_turns or 5,
            example_mode=config.example_mode,
            include_emotion_history=config.include_emotion_history,
        )
        result = CodexResult("", "", False, "not invoked")
        for _ in range(retries + 1):
            result = invoke(prompt, schema_file=schema_file, model=model, timeout=timeout)
            if result.success:
                break
        by_case[case_id] = _record_from_result(case, config, prompt, result, index)
        _write_records(output_file, [by_case[item["id"].strip()] for item in cases if item["id"].strip() in by_case])
    return [by_case[item["id"].strip()] for item in cases]
```

Use the project-configured emotion interval when available, but fall back to `5` without requiring an API key; expose an explicit `--emotion-interval` default of `5` so the CLI run is reproducible.

- [ ] **Step 4: Implement the batch CLI**

```python
parser.add_argument("--dialogues-file", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--schema-file", default="data/config/codex_emotion_result.schema.json")
parser.add_argument("--limit", type=int)
parser.add_argument("--run", action="append", choices=sorted(RUN_CONFIGS))
parser.add_argument("--model")
parser.add_argument("--timeout", type=int, default=180)
parser.add_argument("--retries", type=int, default=1)
parser.add_argument("--emotion-interval", type=int, default=5)
```

Reject non-positive `--limit`, `--timeout`, or `--emotion-interval` before invoking Codex.

- [ ] **Step 5: Run runner tests and existing ablation regression tests**

Run: `./.venv312/bin/python -m pytest tests/test_run_codex_cli_emotion_ablation.py tests/test_run_emotion_ablation.py tests/test_emotion_prompt.py tests/test_evaluate_emotion_ablation.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add scripts/run_codex_cli_emotion_ablation.py tests/test_run_codex_cli_emotion_ablation.py
git commit -m "feat: run resumable codex emotion ablations"
```

### Task 3: Chinese Report Generator

**Files:**
- Create: `scripts/report_codex_cli_emotion_ablation.py`
- Create: `tests/test_report_codex_cli_emotion_ablation.py`

**Interfaces:**
- Consumes: five analysis JSON files, `seed.jsonl`, and `evaluate_records(records, annotations)`.
- Produces: `build_report_data()`, `render_chinese_report()`, `metrics.csv`, `summary.md`, and `report-zh.md`.

- [ ] **Step 1: Add failing aggregate and rendering tests**

```python
def test_build_report_data_includes_global_language_and_context_slices():
    report = build_report_data(RUNS, SEED_RECORDS)
    assert report["runs"]["full"]["overall"]["accuracy"] == 0.5
    assert report["runs"]["full"]["languages"]["zh"]["total"] == 1
    assert report["runs"]["full"]["contexts"]["high"]["total"] == 1


def test_render_chinese_report_contains_metrics_failures_and_limitations():
    text = render_chinese_report(REPORT, metadata={"commit": "abc123", "codex_version": "0.142.4"})
    assert "# Codex CLI 情绪识别消融实验报告" in text
    assert "Macro F1" in text
    assert "调用失败" in text
    assert "组合消融" in text
```

- [ ] **Step 2: Run report tests and confirm missing-module failure**

Run: `./.venv312/bin/python -m pytest tests/test_report_codex_cli_emotion_ablation.py -q`

Expected: FAIL during import because the report module does not exist.

- [ ] **Step 3: Implement sliced evaluation using existing matching logic**

```python
def _slice(records, annotations, predicate):
    selected = [item for item in annotations if predicate(item)]
    selected_ids = {item["case_id"] for item in selected}
    predictions = [item for item in records if item.get("case_id") in selected_ids]
    return evaluate_records(predictions, selected)


def build_report_data(runs, annotations):
    output = {"runs": {}}
    for name, records in runs.items():
        output["runs"][name] = {
            "overall": evaluate_records(records, annotations),
            "languages": {
                language: _slice(records, annotations, lambda item, lang=language: item.get("language") == lang)
                for language in ("zh", "en")
            },
            "contexts": {
                level: _slice(records, annotations, lambda item, value=level: item.get("context_dependency") == value)
                for level in ("none", "low", "medium", "high")
            },
            "failures": sum(record.get("success") is not True for record in records),
        }
    return output
```

Normalize seed records into evaluator annotations with `case_id` and `expected` before slicing.

- [ ] **Step 4: Implement deterministic CSV and Chinese Markdown rendering**

The overall table must contain `run`, `samples`, `valid_predictions`, `failures`, `correct`, `accuracy`, `macro_f1`, `accuracy_delta_vs_full`, and `macro_f1_delta_vs_full`. The report must include overall, language, context-dependency, label-confusion, example-error, methodology, and limitations sections.

- [ ] **Step 5: Run report tests**

Run: `./.venv312/bin/python -m pytest tests/test_report_codex_cli_emotion_ablation.py tests/test_evaluate_emotion_analysis.py tests/test_evaluate_emotion_ablation.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts/report_codex_cli_emotion_ablation.py tests/test_report_codex_cli_emotion_ablation.py
git commit -m "feat: report codex emotion ablation metrics"
```

### Task 4: Ten-Case Pilot Gate

**Files:**
- Generate: `data/records/codex_cli_ablation/pilot/*.json`

**Interfaces:**
- Consumes: runner from Task 2 and `release/seed.jsonl` exported into legacy dialogue format.
- Produces: 50 result records used only as a gate before Task 5.

- [ ] **Step 1: Export the 64 seed records to legacy input format**

Run:

```bash
./.venv312/bin/python scripts/benchmark/export_emotion_ablation_v2.py \
  --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl \
  --output-dir data/records/codex_cli_ablation/input
```

Expected: `dialogues.jsonl` and `labels.jsonl` each contain 64 records.

- [ ] **Step 2: Run one-case `full` smoke test**

Run:

```bash
./.venv312/bin/python scripts/run_codex_cli_emotion_ablation.py \
  --dialogues-file data/records/codex_cli_ablation/input/dialogues.jsonl \
  --output-dir data/records/codex_cli_ablation/smoke \
  --run full --limit 1 --timeout 180 --retries 1
```

Expected: one valid, supported emotion result or a concrete Codex connectivity/rate-limit error recorded for diagnosis.

- [ ] **Step 3: Run the 10-case pilot across all five groups**

Run:

```bash
./.venv312/bin/python scripts/run_codex_cli_emotion_ablation.py \
  --dialogues-file data/records/codex_cli_ablation/input/dialogues.jsonl \
  --output-dir data/records/codex_cli_ablation/pilot \
  --limit 10 --timeout 180 --retries 1
```

Expected: five files with 10 records each.

- [ ] **Step 4: Verify the pilot gate**

Run a validation command that checks 50 total records, unique `(run, case_id)` pairs, and at least 48 successful supported labels. If the gate fails, stop before the formal run and report the exact failure families.

### Task 5: Seed-64 Formal Run and Chinese Report

**Files:**
- Generate: `data/records/codex_cli_ablation/seed64/*.json`
- Generate: `data/records/codex_cli_ablation/seed64/metrics.csv`
- Generate: `data/records/codex_cli_ablation/seed64/summary.md`
- Generate: `data/records/codex_cli_ablation/seed64/report-zh.md`

**Interfaces:**
- Consumes: successful pilot, runner, report generator, 64-record input and labels.
- Produces: final experiment outputs and user-facing report.

- [ ] **Step 1: Run all five groups on all 64 seed records**

Run:

```bash
./.venv312/bin/python scripts/run_codex_cli_emotion_ablation.py \
  --dialogues-file data/records/codex_cli_ablation/input/dialogues.jsonl \
  --output-dir data/records/codex_cli_ablation/seed64 \
  --timeout 180 --retries 1
```

Expected: five JSON files with 64 traceable records each; reruns resume successful pairs.

- [ ] **Step 2: Generate metrics and the Chinese report**

Run:

```bash
./.venv312/bin/python scripts/report_codex_cli_emotion_ablation.py \
  --benchmark-file data/benchmarks/emotion_ablation_v2/release/seed.jsonl \
  --results-dir data/records/codex_cli_ablation/seed64 \
  --output-dir data/records/codex_cli_ablation/seed64
```

Expected: `metrics.csv`, `summary.md`, and `report-zh.md` are created.

- [ ] **Step 3: Verify record and report consistency**

Check that each run has exactly 64 unique case IDs, every seed ID appears once per run, metrics recomputed from JSON equal `metrics.csv`, and all Markdown tables use the same values. Failed predictions must remain in denominators and appear in the failure count.

- [ ] **Step 4: Run the complete focused verification suite**

Run:

```bash
./.venv312/bin/python -m pytest \
  tests/test_run_codex_cli_emotion_ablation.py \
  tests/test_report_codex_cli_emotion_ablation.py \
  tests/test_run_emotion_ablation.py \
  tests/test_evaluate_emotion_analysis.py \
  tests/test_evaluate_emotion_ablation.py \
  tests/test_emotion_prompt.py -q
```

Expected: all tests PASS with zero failures.

- [ ] **Step 5: Inspect final repository state**

Run: `git status --short --branch` and `git diff --check`.

Expected: only intentional code changes, ignored experiment outputs, and pre-existing `.venv/` plus `.venv312/` directories remain; no unrelated files are staged.
