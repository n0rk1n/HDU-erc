# Dataset Card

Emotion Ablation V2 is a bilingual benchmark package for emotion-recognition ablation experiments.

Version `0.1.0` contains a deterministic synthetic generated 500-record formal release plus a 64-record seed set. The formal release is split into `core_parallel`, `extended_independent`, and `challenge` records so experiments can report both headline metrics and harder diagnostic slices.

The seed set remains as a compact reference for annotation style, bilingual pairing, and future expansion examples.

The formal release uses Chinese and English records whose `expected` fields are generator target labels from the supported 32-label taxonomy. They are not independently human-annotated or adjudicated labels. `label_provenance=synthetic_generator_target` records this explicitly.

The zero-byte files under `annotation/` are placeholders for future dual human annotation and adjudication. `annotation_status=released` currently denotes packaging/release state, not human review.
