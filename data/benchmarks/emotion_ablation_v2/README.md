# Emotion Ablation V2

This benchmark package stores bilingual emotion-recognition cases for ablation experiments.

Primary workflow:

1. Add candidates under `raw_candidates/`.
2. Annotate independently under `annotation/`.
3. Adjudicate conflicts.
4. Release reviewed records under `release/`.
5. Validate, summarize, and export records with `scripts/benchmark/`.

Command examples:

```bash
python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl
python scripts/benchmark/check_parallel_equivalence.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl
python scripts/benchmark/summarize_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl --output-dir data/benchmarks/emotion_ablation_v2/reports
python scripts/benchmark/export_emotion_ablation_v2.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl --output-dir data/records/ablation_v2_export
```

Version `0.1.0` is a skeleton for the seed release workflow. Later tasks add records, validation helpers, summaries, and compatibility exports.
