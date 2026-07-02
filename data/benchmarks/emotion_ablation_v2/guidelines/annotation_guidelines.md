# Annotation Guidelines

Annotators assign one `expected` primary emotion from the 32-label set. They may add up to three `secondary_emotions`.

Required annotation fields:

- `expected`
- `secondary_emotions`
- `intensity`
- `ambiguity_level`
- `context_dependency`
- `evidence_span`
- `rationale`

Reject a candidate when the emotion evidence is too weak, the language is unnatural, the sample contains personal identifying information, or a safety-sensitive example includes actionable harm details.

Each candidate should be annotated independently by two annotators before adjudication. If annotators disagree, reviewers should compare the evidence span, rationale, intensity, and any confusable-label boundary before deciding whether to release, revise, or reject the candidate.
