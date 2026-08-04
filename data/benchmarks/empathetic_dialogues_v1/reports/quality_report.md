# Quality Report

- Official source archive SHA-256 verified before conversion.
- Parsed test conversations: 2,542.
- Project-schema validation errors: 0.
- Label coverage: 32 of 32 project labels.
- Balanced seed: 64 records, exactly 2 per label.
- Prefix coverage: the first 32 balanced-seed records cover all 32 labels once.
- Ground-truth alignment: formal records use the first target-speaker utterance;
  later-turn transformations are isolated in `context_diagnostic.jsonl` with a
  weak-alignment flag.
- Few-shot coverage: 64 human-authored examples from the official train split,
  exactly 2 per label and disjoint from test case IDs.
- Placeholder cleanup: `_comma_` restored to commas.
- Leakage control: source situation prompt retained only as provenance and not
  exported into the model dialogue input.
- Formal context distribution: `none=2542`; history ablations are therefore
  declared no-op and are not run on this benchmark.
- The original label/input mismatch is documented in `remediation_report.md`;
  the misaligned predictions are not retained as a model conclusion.

No claim is made that later turns were independently re-annotated or adjudicated
at utterance level. Exact 32-class metrics remain primary; family metrics are
diagnostic only.
