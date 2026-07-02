# Quality Report

This quality report covers the version `0.1.0` formal release and the retained seed reference set.

## Formal Release Checks

- Records: 500
- Splits:
  - `core_parallel`: 256 records, 128 bilingual pairs
  - `extended_independent`: 180 records
  - `challenge`: 64 records
- Languages: 250 English, 250 Chinese
- Label coverage: all 32 labels appear in both languages
- Labels file: `release/labels.jsonl` matches the 500-record formal release order
- Generation command: `python scripts/benchmark/generate_emotion_ablation_v2_release.py`
- Validation commands:
  - `python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl`
  - `python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/extended_independent.jsonl`
  - `python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/challenge.jsonl`
- Parallel check command: `python scripts/benchmark/check_parallel_equivalence.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl`

## Seed Release Checks

- Records: 64
- Languages: 32 English, 32 Chinese
- Label coverage: all 32 labels appear exactly twice
- Parallel seed pairs: 16
- Validation command: `python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl`
- Parallel check command: `python scripts/benchmark/check_parallel_equivalence.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl`

Future reports should add adjudication outcomes and rejection reasons when more raw candidates move through human review.
