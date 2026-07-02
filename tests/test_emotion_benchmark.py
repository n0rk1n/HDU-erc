from pathlib import Path

from chatbot.emotion_labels import EMOTION_LABEL_SET
from scripts.benchmark.emotion_benchmark import load_jsonl
from scripts.benchmark.emotion_benchmark import parallel_equivalence_errors
from scripts.benchmark.emotion_benchmark import validate_record
from scripts.benchmark.emotion_benchmark import validate_records


BENCHMARK_ROOT = Path("data/benchmarks/emotion_ablation_v2")


def load_formal_release_records():
    records = []
    for filename in ["core_parallel.jsonl", "extended_independent.jsonl", "challenge.jsonl"]:
        records.extend(load_jsonl(BENCHMARK_ROOT / "release" / filename))
    return records


def test_benchmark_v2_structure_exists():
    expected_paths = [
        "README.md",
        "metadata.json",
        "schema.json",
        "guidelines/annotation_guidelines.md",
        "guidelines/label_taxonomy.md",
        "guidelines/bilingual_parallel_policy.md",
        "guidelines/quality_control.md",
        "raw_candidates/generated_candidates.jsonl",
        "annotation/annotator_a.jsonl",
        "annotation/annotator_b.jsonl",
        "annotation/adjudication.jsonl",
        "release/core_parallel.jsonl",
        "release/extended_independent.jsonl",
        "release/challenge.jsonl",
        "reports/dataset_card.md",
        "reports/quality_report.md",
    ]

    missing = [
        relative_path
        for relative_path in expected_paths
        if not (BENCHMARK_ROOT / relative_path).exists()
    ]

    assert missing == []


def test_load_jsonl_ignores_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"case_id": "a"}\n\n{"case_id": "b"}\n', encoding="utf-8")

    assert load_jsonl(path) == [{"case_id": "a"}, {"case_id": "b"}]


def test_load_jsonl_reports_line_number_for_bad_json(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"case_id": "a"}\nnot-json\n', encoding="utf-8")

    try:
        load_jsonl(path)
    except ValueError as exc:
        assert f"{path}:2" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_validate_record_accepts_complete_seed_case():
    record = {
        "case_id": "seed-0001-en",
        "language": "en",
        "subset": "seed",
        "seed_group": "independent_seed",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "workplace_interview",
        "annotation_status": "released",
        "secondary_emotions": ["apprehensive"],
        "intensity": 0.7,
        "ambiguity_level": "low",
        "context_dependency": "none",
        "quality_flags": [],
    }

    assert validate_record(record) == []


def test_validate_record_rejects_invalid_label():
    record = {
        "case_id": "seed-0001-en",
        "language": "en",
        "subset": "seed",
        "expected": "worried",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "workplace_interview",
        "annotation_status": "released",
    }

    errors = validate_record(record)

    assert "expected must be one of the supported emotion labels" in errors


def test_validate_record_rejects_non_snake_case_scenario():
    record = {
        "case_id": "seed-0001-en",
        "language": "en",
        "subset": "seed",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "Academic Presentation",
        "annotation_status": "released",
    }

    errors = validate_record(record)

    assert "scenario must be lowercase snake case" in errors


def test_validate_record_rejects_release_stage_candidate_status():
    record = {
        "case_id": "seed-0001-en",
        "language": "en",
        "subset": "seed",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "workplace_interview",
        "annotation_status": "candidate",
        "source_stage": "release",
    }

    errors = validate_record(record)

    assert "release records must be adjudicated or released" in errors


def test_validate_record_rejects_boolean_intensity():
    record = {
        "case_id": "seed-0001-en",
        "language": "en",
        "subset": "seed",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "workplace_interview",
        "annotation_status": "released",
        "intensity": True,
    }

    errors = validate_record(record)

    assert "intensity must be a number from 0.0 to 1.0" in errors


def test_validate_records_rejects_duplicate_case_ids():
    records = [
        {
            "case_id": "seed-0001-en",
            "language": "en",
            "subset": "seed",
            "expected": "anxious",
            "turn_count": 1,
            "history": [],
            "current_input": "I keep replaying tomorrow's interview in my head.",
            "scenario": "workplace_interview",
            "annotation_status": "released",
        },
        {
            "case_id": "seed-0001-en",
            "language": "zh",
            "subset": "seed",
            "expected": "anxious",
            "turn_count": 1,
            "history": [],
            "current_input": "我一直在想明天面试的事。",
            "scenario": "workplace_interview",
            "annotation_status": "released",
        },
    ]

    errors = validate_records(records)

    assert "duplicate case_id: seed-0001-en" in errors


def test_parallel_equivalence_rejects_context_dependency_gap():
    records = [
        {
            "case_id": "pair-1-en",
            "pair_id": "pair-1",
            "language": "en",
            "subset": "core_parallel",
            "expected": "anxious",
            "turn_count": 1,
            "history": [],
            "current_input": "I keep replaying tomorrow's interview in my head.",
            "scenario": "workplace_interview",
            "annotation_status": "released",
            "context_dependency": "none",
        },
        {
            "case_id": "pair-1-zh",
            "pair_id": "pair-1",
            "language": "zh",
            "subset": "core_parallel",
            "expected": "anxious",
            "turn_count": 1,
            "history": [],
            "current_input": "我一直在想明天面试的事。",
            "scenario": "workplace_interview",
            "annotation_status": "released",
            "context_dependency": "high",
        },
    ]

    errors = parallel_equivalence_errors(records)

    assert "pair-1: context_dependency differs by more than one level" in errors


def test_seed_release_covers_all_supported_labels():
    records = load_jsonl(BENCHMARK_ROOT / "release" / "seed.jsonl")
    labels = {record["expected"] for record in records}

    assert len(records) == 64
    assert labels == EMOTION_LABEL_SET


def test_seed_release_has_expected_language_mix():
    records = load_jsonl(BENCHMARK_ROOT / "release" / "seed.jsonl")
    language_counts = {}
    for record in records:
        language_counts[record["language"]] = language_counts.get(record["language"], 0) + 1

    assert language_counts == {"en": 32, "zh": 32}


def test_seed_release_validates_without_errors():
    records = load_jsonl(BENCHMARK_ROOT / "release" / "seed.jsonl")

    assert validate_records(records) == []


def test_formal_release_has_500_records_across_splits():
    assert len(load_jsonl(BENCHMARK_ROOT / "release" / "core_parallel.jsonl")) == 256
    assert len(load_jsonl(BENCHMARK_ROOT / "release" / "extended_independent.jsonl")) == 180
    assert len(load_jsonl(BENCHMARK_ROOT / "release" / "challenge.jsonl")) == 64
    assert len(load_formal_release_records()) == 500


def test_formal_release_validates_without_errors():
    assert validate_records(load_formal_release_records()) == []


def test_formal_release_covers_labels_in_both_languages():
    records = load_formal_release_records()
    language_counts = {}
    labels_by_language = {"en": set(), "zh": set()}
    for record in records:
        language_counts[record["language"]] = language_counts.get(record["language"], 0) + 1
        labels_by_language[record["language"]].add(record["expected"])

    assert language_counts == {"en": 250, "zh": 250}
    assert labels_by_language == {"en": EMOTION_LABEL_SET, "zh": EMOTION_LABEL_SET}


def test_formal_release_labels_match_all_release_cases():
    records = load_formal_release_records()
    labels = load_jsonl(BENCHMARK_ROOT / "release" / "labels.jsonl")

    assert labels == [
        {"id": record["case_id"], "expected": record["expected"]}
        for record in records
    ]


def test_core_parallel_release_has_128_bilingual_pairs():
    records = load_jsonl(BENCHMARK_ROOT / "release" / "core_parallel.jsonl")
    pair_ids = {record["pair_id"] for record in records}

    assert len(pair_ids) == 128
    assert parallel_equivalence_errors(records) == []
