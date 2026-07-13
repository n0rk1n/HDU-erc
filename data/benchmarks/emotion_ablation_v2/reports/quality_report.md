# Quality Report

This quality report covers the deterministic synthetic version `0.1.0` formal release and the retained seed reference set.

## Formal Release Checks

- Records: 500
- Splits: `core_parallel=256`, `extended_independent=180`, `challenge=64`
- Languages: 250 English, 250 Chinese
- Label coverage: all 32 generator target labels appear in both languages
- Label provenance: `synthetic_generator_target`; `expected` is the generator target, not an independently annotated ground truth label
- Annotation/adjudication files: zero-byte placeholders reserved for future human dual annotation and adjudication
- `annotation_status=released`: packaging state only; it does not assert human review
- Generation command: `python scripts/benchmark/generate_emotion_ablation_v2_release.py`

## Seed Release Checks

- Records: 64
- Languages: 32 English, 32 Chinese
- Label coverage: all 32 labels appear exactly twice
- Parallel seed pairs: 16

Human dual annotation, adjudication, agreement statistics, and rejection-reason reporting remain future work.
