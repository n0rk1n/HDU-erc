# Emotion Ablation V2 Benchmark Design

## Summary

Build `emotion_ablation_v2`, a complete bilingual emotion-recognition benchmark for the chatbot's ablation experiments. The benchmark expands the current 6-case JSONL sample into a governed dataset system with candidate generation, bilingual parallel cases, independent natural-language cases, challenge cases, dual annotation, adjudication, quality reports, schema validation, compatibility export, and layered evaluation.

This design favors completeness over minimum viable scope. The first implementation should create the full directory structure, documentation, schema, validation/export tooling, and a high-quality seed set. The full dataset can then scale to 600+ released cases without changing the core format.

## Current Context

The repository currently has:

- `data/examples/ablation_dialogues.jsonl`: 6 English dialogue cases.
- `data/examples/ablation_labels.jsonl`: 6 matching expected labels.
- `scripts/run_emotion_ablation.py`: runs five ablation configurations over JSONL dialogues.
- `scripts/evaluate_emotion_ablation.py`: compares run outputs against labels.
- `chatbot/emotion_labels.py`: 32 supported emotion labels.

The current sample is useful for smoke testing, but it is too small for serious analysis. It covers 6 labels and mostly short, direct English examples.

## Goals

1. Expand the ablation data into a reproducible benchmark with clear provenance and release stages.
2. Support Chinese and English evaluation.
3. Preserve cross-language comparability through a bilingual parallel core set.
4. Add natural independent Chinese and English samples for realistic coverage.
5. Add challenge cases for ambiguous, implicit, multi-turn, culturally specific, and safety-sensitive emotion recognition.
6. Cover all 32 supported emotion labels with tiered sample allocation.
7. Keep compatibility with the existing ablation runner and evaluator through an export step.
8. Provide quality-control documentation and tools so dataset changes are reviewable.

## Non-Goals

1. Do not replace the existing `data/examples` smoke-test files in the first implementation.
2. Do not require a hosted database or external annotation platform.
3. Do not ingest raw private user conversations.
4. Do not introduce new emotion labels outside the existing 32-label set.
5. Do not change the chatbot runtime emotion prompt as part of the dataset design.

## Dataset Scope

Target full release:

| Subset | Target Size | Purpose |
| --- | ---: | --- |
| `core_parallel` | About 256 bilingual pairs, about 512 language instances | Main cross-language comparable evaluation set |
| `extended_independent` | About 400-600 cases | Natural Chinese and English coverage without one-to-one pairing |
| `challenge` | About 100-150 cases | Hard cases: ambiguity, irony, implicit evidence, multi-turn dependence, emotion shifts |
| `seed` | About 64 language instances in the first implementation | Reference examples for the schema, annotation policy, and expansion style |
| `legacy_compat` | Current 6 cases migrated into v2 form | Traceability to the current project examples |

The first implementation should deliver a seed set, not the full 600+ case dataset.

## Label Coverage

Use all labels from `chatbot/emotion_labels.py`:

```text
surprised, excited, annoyed, proud, angry, sad, grateful, lonely,
impressed, afraid, disgusted, confident, terrified, hopeful, anxious,
disappointed, joyful, prepared, guilty, furious, nostalgic, jealous,
anticipating, embarrassed, content, devastated, sentimental, caring,
trusting, ashamed, apprehensive, faithful
```

Use tiered allocation instead of perfectly uniform distribution.

### Core and Safety-Relevant Labels

These labels should receive more samples because they are common, important in supportive chat, or safety-adjacent:

```text
anxious, sad, lonely, afraid, terrified, devastated, angry,
disappointed, grateful, hopeful, joyful, content
```

### Social and Self-Evaluation Labels

These labels should receive moderate coverage:

```text
proud, guilty, ashamed, embarrassed, jealous, nostalgic,
sentimental, caring, trusting
```

### Fine-Grained and Boundary Labels

These labels should appear in regular cases and be emphasized in challenge cases:

```text
apprehensive, anticipating, prepared, confident, faithful,
impressed, surprised, annoyed, furious, disgusted, excited
```

## Scenario Coverage

Each label should appear across multiple scenario families where natural. Scenario names should be lowercase snake case.

Recommended scenario families:

| Family | Examples |
| --- | --- |
| `academic` | exams, presentations, thesis defense, course selection, grades |
| `workplace` | interviews, deadlines, performance feedback, teamwork, layoffs |
| `relationships` | friendship drift, family pressure, romantic conflict, misunderstanding, support |
| `self_evaluation` | shame, pride, guilt, confidence, preparation, failure |
| `daily_life` | moving, repairs, money pressure, quiet nights, routines |
| `health` | test results, appointments, waiting for diagnosis, recovery |
| `safety_support` | intense distress, fear, hopelessness, need for immediate support |
| `positive_events` | gratitude, hope, joy, trust, contentment, excitement |
| `complex_expression` | irony, restraint, mixed emotion, emotion shift, indirect evidence |

## Directory Structure

Create a new dataset package:

```text
data/benchmarks/emotion_ablation_v2/
  README.md
  metadata.json
  schema.json

  guidelines/
    annotation_guidelines.md
    label_taxonomy.md
    bilingual_parallel_policy.md
    quality_control.md

  raw_candidates/
    generated_candidates.jsonl
    imported_legacy_candidates.jsonl

  annotation/
    annotator_a.jsonl
    annotator_b.jsonl
    adjudication.jsonl

  release/
    core_parallel.jsonl
    extended_independent.jsonl
    challenge.jsonl
    seed.jsonl
    labels.jsonl

  reports/
    dataset_card.md
    quality_report.md
    label_distribution.csv
    scenario_distribution.csv
```

## Dataset Lifecycle

### 1. Candidate Generation

Candidates may come from:

- Human-authored scenario matrices.
- LLM-assisted candidate generation.
- Migration of existing legacy examples.
- Highly abstracted realistic examples with no identifiable personal information.

All candidates start in `raw_candidates/` and must record source information. No generated candidate can move directly to `release/`.

### 2. Dual Annotation

Each candidate should be annotated independently by at least two annotators. Store results in:

- `annotation/annotator_a.jsonl`
- `annotation/annotator_b.jsonl`

Each annotation includes primary emotion, secondary emotions, intensity, ambiguity, context dependency, evidence span, rationale, and optional rejection reason.

### 3. Adjudication

When annotators disagree:

- If labels are in a known confusable group, adjudicate and record `conflict_type: "confusable_boundary"`.
- If labels are unrelated, review the text for weak evidence or unclear wording.
- If evidence remains unclear, reject or rewrite the candidate.
- Safety-sensitive samples require secondary review even when annotators agree.

Store final decisions in `annotation/adjudication.jsonl`.

### 4. Quality Filtering

Before release, reject or revise cases with:

- Invalid emotion labels.
- Unnatural Chinese or English.
- Weak emotion evidence.
- Overly explicit or template-like phrasing.
- Parallel bilingual mismatch.
- Unsafe details in crisis or self-harm-adjacent examples.
- Personal or identifiable information.

### 5. Release Splitting

Released cases go into:

- `release/core_parallel.jsonl`
- `release/extended_independent.jsonl`
- `release/challenge.jsonl`
- `release/seed.jsonl`
- `release/labels.jsonl`

### 6. Compatibility Export

An export script should convert released v2 cases into existing ablation format:

- `dialogues.jsonl`: fields accepted by `scripts/run_emotion_ablation.py`.
- `labels.jsonl`: fields accepted by `scripts/evaluate_emotion_ablation.py`.

This keeps the old runner usable while allowing v2 to contain richer metadata.

## Case Schema

Each released case should use this logical schema:

```json
{
  "case_id": "core-0001-en",
  "pair_id": "core-0001",
  "language": "en",
  "subset": "core_parallel",
  "target_emotion": "anxious",
  "expected": "anxious",
  "secondary_emotions": ["apprehensive"],
  "intensity": 0.78,
  "ambiguity_level": "low",
  "scenario": "academic_presentation",
  "context_dependency": "medium",
  "turn_count": 3,
  "history": [
    {"role": "human", "content": "I rehearsed twice tonight."},
    {"role": "ai", "content": "That sounds like careful preparation."}
  ],
  "current_input": "My hands are still shaking when I imagine standing in front of everyone.",
  "evidence_span": "My hands are still shaking",
  "rationale": "The user expresses physiological anxiety about a future presentation.",
  "quality_flags": [],
  "annotation_status": "adjudicated",
  "source_stage": "release"
}
```

### Required Fields

- `case_id`: unique stable ID.
- `language`: `zh` or `en`.
- `subset`: `core_parallel`, `extended_independent`, `challenge`, `seed`, or `legacy_compat`.
- `expected`: one of the 32 supported emotion labels.
- `turn_count`: positive integer.
- `history`: list of `{role, content}` objects, with role `human` or `ai`.
- `current_input`: non-empty string.
- `scenario`: lowercase snake case.
- `annotation_status`: `candidate`, `annotated`, `adjudicated`, `released`, or `rejected`.

### Optional but Recommended Fields

- `pair_id`: required for `core_parallel`, optional otherwise.
- `target_emotion`: intended generation label.
- `secondary_emotions`: list of 0-3 labels.
- `intensity`: number from 0.0 to 1.0.
- `ambiguity_level`: `low`, `medium`, or `high`.
- `context_dependency`: `none`, `low`, `medium`, or `high`.
- `evidence_span`: short supporting phrase.
- `rationale`: one-sentence annotation explanation.
- `quality_flags`: list of known flags.
- `source_stage`: `raw`, `annotation`, or `release`.

## Annotation Guidelines

Every label definition should include:

- English label.
- Chinese explanation.
- Positive criteria.
- Negative criteria.
- Common confusions.
- English examples.
- Chinese examples.

Example:

```text
label: anxious
中文解释：对未来不确定结果的担忧、紧张或预期性压力。
判定依据：用户表达担心、紧张、坐立不安、害怕失败，但危险感不一定迫近。
不要误标为：
- afraid：更偏具体威胁或害怕某事发生。
- apprehensive：更轻、更犹豫、更不确定。
- terrified：强烈恐惧，接近惊恐。
```

## Confusable Label Groups

Document these groups in `guidelines/label_taxonomy.md`:

| Group | Distinction |
| --- | --- |
| `anxious` / `apprehensive` / `afraid` / `terrified` | anticipatory worry, mild unease, concrete fear, intense fear |
| `sad` / `lonely` / `devastated` | general low mood, lack of connection, severe collapse |
| `angry` / `annoyed` / `furious` / `disgusted` | displeasure, irritation, rage, revulsion |
| `grateful` / `caring` / `trusting` / `faithful` | receiving kindness, concern for others, confidence in someone, committed belief |
| `confident` / `prepared` / `proud` | ability certainty, readiness, achievement-based self-regard |
| `hopeful` / `anticipating` / `excited` | positive outlook, awaiting an event, high-energy eagerness |
| `guilty` / `ashamed` / `embarrassed` | responsibility for harm, self-worth shame, social awkwardness |

## Bilingual Parallel Policy

For `core_parallel`:

- Chinese and English instances share a `pair_id`.
- They must preserve scenario, user intent, emotional evidence, intensity, and context dependency.
- They do not need to be literal translations.
- They should be natural in each language.
- If direct translation changes emotion strength or cultural implication, rewrite both sides.

Parallel checks:

- Primary label must match.
- Intensity difference should be no more than 0.15.
- Context dependency should match or be adjacent.
- Evidence spans should correspond semantically.
- History length and turn role sequence should match.

## Quality Flags

Use these known flags:

```text
too_template_like
emotion_too_explicit
emotion_evidence_weak
parallel_mismatch
label_boundary_case
safety_sensitive
requires_context
contains_irony
mixed_emotion
cultural_specificity
```

Flags are not automatically defects. Some are useful for challenge or diagnostic evaluation.

## Seed Dataset Strategy

The first implementation should create a high-density seed set of about 64 language instances:

- 16 bilingual seed pairs using `subset: "seed"` and a `seed_group` value of `core_parallel_seed`, for 32 language instances.
- 16 Chinese cases and 16 English cases using `subset: "seed"` and a `seed_group` value of `independent_seed`.
- Cover all 32 labels at least once as `expected`.
- Include full metadata fields for every seed case.
- Include at least several `challenge`-style flags in the seed set.

Seed cases should be models for later expansion. They should avoid repetitive templates, excessive explicit emotion words, and narrow scenario coverage.

Expansion constraints for the full dataset:

- At least 40% of cases should not directly state the emotion label.
- At least 30% should require context for the best answer.
- Each label should appear in at least 3 scenario families where natural.
- Each label should have at least 2 boundary or challenge cases.
- Chinese cases must sound like natural Chinese chat.
- English cases must include natural conversational phrasing.
- Safety-sensitive examples must avoid actionable harm details.

## Tooling

Add lightweight scripts under `scripts/benchmark/`:

```text
scripts/benchmark/
  validate_emotion_benchmark.py
  export_emotion_ablation_v2.py
  summarize_emotion_benchmark.py
  check_parallel_equivalence.py
```

### `validate_emotion_benchmark.py`

Validate:

- JSONL syntax.
- Required fields.
- Label membership.
- ID uniqueness.
- Valid roles.
- Valid enum values.
- Intensity range.
- Release cases have `annotation_status` of `adjudicated` or `released`.
- `core_parallel` cases have valid `pair_id`.

### `export_emotion_ablation_v2.py`

Export selected release subsets to:

- `dialogues.jsonl`
- `labels.jsonl`

The export format should remain compatible with current ablation scripts.

### `summarize_emotion_benchmark.py`

Generate:

- Label distribution.
- Language distribution.
- Scenario distribution.
- Subset counts.
- Ambiguity distribution.
- Context dependency distribution.
- Quality flag distribution.

### `check_parallel_equivalence.py`

Check:

- Every `core_parallel` `pair_id` has one `zh` and one `en` case.
- Expected labels match.
- Intensity difference is within threshold.
- Context dependency values are compatible.
- History role sequences match.

## Evaluation Design

Continue using:

```bash
python scripts/run_emotion_ablation.py
python scripts/evaluate_emotion_ablation.py
```

Add v2 evaluation summaries by:

- `language`
- `subset`
- `scenario`
- `context_dependency`
- `ambiguity_level`
- `quality_flags`
- `emotion_group`
- `pair_id`

Reports should include:

1. Main table: overall accuracy and macro F1 per ablation run.
2. Stratified tables: metrics by language, scenario, subset, difficulty, and context dependency.
3. Diagnostic tables: common confusion pairs.
4. Parallel consistency table: whether the same model/run behaves differently on paired Chinese and English cases.

## Acceptance Criteria

The implementation is complete when:

1. The v2 directory structure exists.
2. Documentation files explain dataset scope, label taxonomy, bilingual policy, and quality control.
3. `schema.json` defines the released case format.
4. A seed JSONL file exists with representative Chinese and English cases.
5. Legacy examples are represented or importable in v2 format.
6. Validation tooling catches malformed records and invalid labels.
7. Export tooling produces `dialogues.jsonl` and `labels.jsonl` compatible with existing scripts.
8. Summary tooling reports distributions.
9. Parallel-check tooling validates bilingual pairs.
10. Tests cover validation, export, summary, and parallel checks.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Dataset becomes too complex to maintain | Keep JSONL files plain and scripts lightweight |
| Labels are hard to distinguish | Write explicit confusable-label rules and require rationales |
| Bilingual pairs become unnatural | Use semantic equivalence, not literal translation |
| Seed data becomes template-like | Enforce scenario and expression diversity |
| Challenge cases distort headline scores | Report challenge separately from main core metrics |
| Safety-sensitive examples become too explicit | Require safety review and avoid actionable details |
| Existing scripts cannot consume v2 directly | Provide compatibility export |

## Implementation Phases

### Phase 1: Structure and Specification

- Create benchmark directory.
- Add README, metadata, schema, and guidelines.
- Add seed data plan.

### Phase 2: Seed Data

- Add about 64 seed language instances.
- Cover all 32 labels.
- Include bilingual parallel and independent cases.

### Phase 3: Tooling

- Add validation script.
- Add compatibility export script.
- Add summary script.
- Add parallel-equivalence script.
- Add tests.

### Phase 4: Evaluation Integration

- Document how to export v2 and run existing ablation scripts.
- Add stratified summary support if needed.

### Phase 5: Full Expansion

- Generate or author additional candidates.
- Run annotation, adjudication, and quality filtering.
- Release full benchmark splits and quality reports.
