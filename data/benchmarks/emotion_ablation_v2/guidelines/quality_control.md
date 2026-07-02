# Quality Control

Known quality flags:

- `too_template_like`
- `emotion_too_explicit`
- `emotion_evidence_weak`
- `parallel_mismatch`
- `label_boundary_case`
- `safety_sensitive`
- `requires_context`
- `contains_irony`
- `mixed_emotion`
- `cultural_specificity`

Reject or revise records with invalid labels, unnatural language, weak emotion evidence, overly explicit phrasing, parallel bilingual mismatch, unsafe actionable details, or personal identifying information.

Safety-sensitive samples require secondary review even when annotators agree. Confusable boundary cases should include a rationale that explains why the selected primary label is stronger than nearby labels.
