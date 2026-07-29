from pathlib import Path

from scripts.benchmark.emotion_benchmark import export_label
from scripts.benchmark.emotion_benchmark import load_jsonl
from scripts.benchmark.emotion_benchmark import validate_record
from scripts.benchmark.emotion_benchmark import validate_records


BENCHMARK_ROOT = Path("data/benchmarks/empathetic_dialogues_v1")


def test_public_benchmark_structure_exists():
    expected_paths = [
        "README.md",
        "metadata.json",
        "LICENSE-NOTICE.md",
        "release/test.jsonl",
        "release/balanced_seed.jsonl",
        "release/context_diagnostic.jsonl",
        "few_shot/train_examples.jsonl",
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


def test_export_label_retains_public_label_provenance():
    record = {
        "case_id": "ed-test-1",
        "expected": "anxious",
        "label_provenance": "human_authored_emotion_grounding",
    }

    assert export_label(record) == {
        "id": "ed-test-1",
        "expected": "anxious",
        "label_provenance": "human_authored_emotion_grounding",
    }


def test_validate_record_accepts_public_benchmark_case():
    record = {
        "case_id": "ed-test-1",
        "language": "en",
        "subset": "empathetic_dialogues_test",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "open_domain_dialogue",
        "annotation_status": "released",
        "source_stage": "release",
        "label_provenance": "human_authored_emotion_grounding",
        "context_dependency": "none",
        "quality_flags": [],
    }

    assert validate_record(record) == []


def test_validate_record_rejects_invalid_label():
    record = {
        "case_id": "ed-test-1",
        "language": "en",
        "subset": "empathetic_dialogues_test",
        "expected": "worried",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "open_domain_dialogue",
        "annotation_status": "released",
        "label_provenance": "human_authored_emotion_grounding",
    }

    errors = validate_record(record)

    assert "expected must be one of the supported emotion labels" in errors


def test_validate_record_rejects_non_snake_case_scenario():
    record = {
        "case_id": "ed-test-1",
        "language": "en",
        "subset": "empathetic_dialogues_test",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "Open Domain Dialogue",
        "annotation_status": "released",
        "label_provenance": "human_authored_emotion_grounding",
    }

    errors = validate_record(record)

    assert "scenario must be lowercase snake case" in errors


def test_validate_record_rejects_release_stage_candidate_status():
    record = {
        "case_id": "ed-test-1",
        "language": "en",
        "subset": "empathetic_dialogues_test",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "open_domain_dialogue",
        "annotation_status": "candidate",
        "source_stage": "release",
        "label_provenance": "human_authored_emotion_grounding",
    }

    errors = validate_record(record)

    assert "release packaging records must use adjudicated or released status" in errors


def test_validate_record_rejects_public_benchmark_without_label_provenance():
    record = {
        "case_id": "ed-test-1",
        "language": "en",
        "subset": "empathetic_dialogues_test",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep worrying about tomorrow.",
        "scenario": "open_domain_dialogue",
        "annotation_status": "released",
        "source_stage": "release",
    }

    errors = validate_record(record)

    assert any(
        error.startswith("label_provenance must be one of") for error in errors
    )


def test_validate_record_rejects_boolean_intensity():
    record = {
        "case_id": "ed-test-1",
        "language": "en",
        "subset": "empathetic_dialogues_test",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "open_domain_dialogue",
        "annotation_status": "released",
        "label_provenance": "human_authored_emotion_grounding",
        "intensity": True,
    }

    errors = validate_record(record)

    assert "intensity must be a number from 0.0 to 1.0" in errors


def test_validate_records_rejects_duplicate_case_ids():
    record = {
        "case_id": "ed-test-1",
        "language": "en",
        "subset": "empathetic_dialogues_test",
        "expected": "anxious",
        "turn_count": 1,
        "history": [],
        "current_input": "I keep replaying tomorrow's interview in my head.",
        "scenario": "open_domain_dialogue",
        "annotation_status": "released",
        "label_provenance": "human_authored_emotion_grounding",
    }

    errors = validate_records([record, dict(record)])

    assert "duplicate case_id: ed-test-1" in errors


def test_public_release_validates_without_errors():
    records = load_jsonl(BENCHMARK_ROOT / "release" / "test.jsonl")

    assert len(records) == 2542
    assert validate_records(records) == []
