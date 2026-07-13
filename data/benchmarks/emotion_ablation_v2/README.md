# Emotion Ablation V2

This benchmark package stores bilingual emotion-recognition cases for ablation experiments.

Current version `0.1.0` is a deterministic synthetic generated benchmark. Its 500
formal-release `expected` values are generator target labels
(`label_provenance=synthetic_generator_target`), not independent human annotations
or adjudicated ground truth.

Future human-annotation workflow:

1. Add candidates under `raw_candidates/`.
2. Annotate independently under `annotation/`.
3. Adjudicate conflicts.
4. Release human-reviewed records under a future human-labeled release.
5. Validate, summarize, and export records with `scripts/benchmark/`.

The benchmark scripts require Python 3.10 or newer because they use modern type syntax. Run the commands with the active project Python environment.

Version `0.1.0` includes:

- 500 formal release records:
  - `release/core_parallel.jsonl`: 256 records, 128 bilingual pairs.
  - `release/extended_independent.jsonl`: 180 independent natural-language records.
  - `release/challenge.jsonl`: 64 difficult records.
- 64 seed records in `release/seed.jsonl` for annotation and expansion examples.
- `release/labels.jsonl` matching the 500-record formal release order.
- Validation, summary, parallel-check, export, and deterministic generation helpers.

The zero-byte files under `annotation/` are placeholders. No dual annotation or
adjudication has been performed for the 500 generated records. In this version,
`annotation_status=released` means packaged in the release directory only; it does
not imply human review.

Regenerate the formal release:

```bash
python scripts/benchmark/generate_emotion_ablation_v2_release.py
```

Command examples:

```bash
python scripts/benchmark/validate_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl
python scripts/benchmark/check_parallel_equivalence.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl
python scripts/benchmark/summarize_emotion_benchmark.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl --output-dir data/benchmarks/emotion_ablation_v2/reports
python scripts/benchmark/export_emotion_ablation_v2.py --input data/benchmarks/emotion_ablation_v2/release/core_parallel.jsonl --output-dir data/records/ablation_v2_export
```
