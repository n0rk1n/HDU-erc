# Emotion Ablation V2

This benchmark package stores bilingual emotion-recognition cases for ablation experiments.

Primary workflow:

1. Add candidates under `raw_candidates/`.
2. Annotate independently under `annotation/`.
3. Adjudicate conflicts.
4. Release reviewed records under `release/`.
5. Validate, summarize, and export records with `scripts/benchmark/`.

Version `0.1.0` is a skeleton for the seed release workflow. Later tasks add records, validation helpers, summaries, and compatibility exports.
