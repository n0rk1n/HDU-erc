# Emotion Ablation V2

This benchmark package stores bilingual emotion-recognition cases for ablation experiments.

Primary workflow:

1. Add candidates under `raw_candidates/`.
2. Annotate independently under `annotation/`.
3. Adjudicate conflicts.
4. Release reviewed records under `release/`.
5. Validate, summarize, and export records with `scripts/benchmark/`.

The benchmark scripts require Python 3.10 or newer because they use modern type syntax. Run the commands with the active project Python environment.

Command examples:

```bash
python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl
python scripts/benchmark/check_parallel_equivalence.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl
python scripts/benchmark/summarize_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl --output-dir data/benchmarks/emotion_ablation_v2/reports
python scripts/benchmark/export_emotion_ablation_v2.py --input data/benchmarks/emotion_ablation_v2/release/seed.jsonl --output-dir data/records/ablation_v2_export
```

Version `0.1.0` includes a 64-record seed release plus validation, summary, parallel-check, and export helpers.
