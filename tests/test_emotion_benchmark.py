import json
from pathlib import Path

from scripts.benchmark.emotion_benchmark import BenchmarkValidationError
from scripts.benchmark.emotion_benchmark import load_jsonl
from scripts.benchmark.emotion_benchmark import validate_record
from scripts.benchmark.emotion_benchmark import validate_records


BENCHMARK_ROOT = Path("data/benchmarks/emotion_ablation_v2")


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
