# Emotion Ablation V2 500-Record Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `emotion_ablation_v2` from a 64-record seed release to a 500-record formal release while preserving the seed set as reference examples.

**Architecture:** Keep the 64-record `release/seed.jsonl` unchanged as the reference seed. Populate the formal release splits with `core_parallel.jsonl` (256 records), `extended_independent.jsonl` (180 records), and `challenge.jsonl` (64 records), then regenerate `release/labels.jsonl` and distribution reports from those 500 formal records. Add a deterministic generator script so future updates are reproducible.

**Tech Stack:** Python standard library, JSONL, CSV, pytest, existing benchmark helper functions.

---

### Task 1: Formal Release Tests

**Files:**
- Modify: `tests/test_emotion_benchmark.py`

- [ ] **Step 1: Add tests for formal release counts**

Add helper:

```python
def load_formal_release_records():
    records = []
    for filename in ["core_parallel.jsonl", "extended_independent.jsonl", "challenge.jsonl"]:
        records.extend(load_jsonl(BENCHMARK_ROOT / "release" / filename))
    return records
```

Add tests:

```python
def test_formal_release_has_500_records_across_splits():
    assert len(load_jsonl(BENCHMARK_ROOT / "release" / "core_parallel.jsonl")) == 256
    assert len(load_jsonl(BENCHMARK_ROOT / "release" / "extended_independent.jsonl")) == 180
    assert len(load_jsonl(BENCHMARK_ROOT / "release" / "challenge.jsonl")) == 64
    assert len(load_formal_release_records()) == 500


def test_formal_release_validates_without_errors():
    assert validate_records(load_formal_release_records()) == []


def test_formal_release_labels_match_all_release_cases():
    records = load_formal_release_records()
    labels = load_jsonl(BENCHMARK_ROOT / "release" / "labels.jsonl")
    assert labels == [{"id": record["case_id"], "expected": record["expected"]} for record in records]


def test_core_parallel_release_has_128_bilingual_pairs():
    records = load_jsonl(BENCHMARK_ROOT / "release" / "core_parallel.jsonl")
    pair_ids = {record["pair_id"] for record in records}
    assert len(pair_ids) == 128
    assert parallel_equivalence_errors(records) == []
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
.venv312/bin/python -m pytest tests/test_emotion_benchmark.py -k "formal_release or core_parallel_release" -v
```

Expected: fail because formal release split files are empty and labels still target seed.

### Task 2: Deterministic Release Generator

**Files:**
- Create: `scripts/benchmark/generate_emotion_ablation_v2_release.py`
- Modify generated data under `data/benchmarks/emotion_ablation_v2/release/`
- Modify reports under `data/benchmarks/emotion_ablation_v2/reports/`

- [ ] **Step 1: Implement generator**

Create a deterministic Python script that builds:

- 128 pair IDs in `core_parallel.jsonl`, each with one English and one Chinese record.
- 180 independent records in `extended_independent.jsonl`, balanced by language as 90 English and 90 Chinese.
- 64 challenge records in `challenge.jsonl`, balanced by language as 32 English and 32 Chinese.
- `labels.jsonl` from the formal release order: core, extended, challenge.
- `label_distribution.csv` and `scenario_distribution.csv` from the 500 formal records.

Every record must include all benchmark metadata fields already used by seed records.

- [ ] **Step 2: Run generator**

Run:

```bash
.venv312/bin/python scripts/benchmark/generate_emotion_ablation_v2_release.py
```

Expected: writes 500 formal records and reports.

- [ ] **Step 3: Run formal release tests**

Run:

```bash
.venv312/bin/python -m pytest tests/test_emotion_benchmark.py -k "formal_release or core_parallel_release" -v
```

Expected: pass.

### Task 3: Documentation and Verification

**Files:**
- Modify: `data/benchmarks/emotion_ablation_v2/README.md`
- Modify: `data/benchmarks/emotion_ablation_v2/reports/dataset_card.md`
- Modify: `data/benchmarks/emotion_ablation_v2/reports/quality_report.md`
- Modify: `README.md` if needed

- [ ] **Step 1: Update docs**

Docs must state:

- Formal release contains 500 records.
- Splits are `core_parallel=256`, `extended_independent=180`, `challenge=64`.
- Seed remains 64 reference records.
- `labels.jsonl` now corresponds to the 500-record formal release.
- Generation command is `python scripts/benchmark/generate_emotion_ablation_v2_release.py`.

- [ ] **Step 2: Run verification**

Run:

```bash
.venv312/bin/python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl
.venv312/bin/python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/extended_independent.jsonl
.venv312/bin/python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/challenge.jsonl
.venv312/bin/python scripts/benchmark/check_parallel_equivalence.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl
.venv312/bin/python scripts/benchmark/summarize_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl --output-dir /private/tmp/emotion_ablation_v2_summary_check
.venv312/bin/python scripts/benchmark/export_emotion_ablation_v2.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl --output-dir /private/tmp/emotion_ablation_v2_export_check
.venv312/bin/python -m pytest tests/test_emotion_benchmark.py tests/test_emotion_benchmark_cli.py -v
```

Expected: all commands pass.

- [ ] **Step 3: Commit**

```bash
git add README.md data/benchmarks/emotion_ablation_v2 scripts/benchmark tests/test_emotion_benchmark.py docs/superpowers/plans/2026-07-02-emotion-ablation-v2-500-release.md
git commit -m "data: expand emotion benchmark release to 500"
```

