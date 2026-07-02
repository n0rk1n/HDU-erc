# Quality Report

The first quality report covers the seed release and will be regenerated as the dataset expands.

## Seed Release Checks

- Records: 64
- Languages: 32 English, 32 Chinese
- Label coverage: all 32 labels appear exactly twice
- Parallel seed pairs: 16
- Validation command: `python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl`
- Parallel check command: `python scripts/benchmark/check_parallel_equivalence.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl`

Future reports should summarize label distribution, scenario distribution, bilingual parallel checks, adjudication outcomes, rejection reasons, and active quality flags.
