import subprocess
import sys

from scripts.evaluate_emotion_analysis import evaluate_records


def test_evaluate_records_matches_by_turn_count():
    analysis_records = [
        {"turn_count": 1, "emotion": "anxious", "success": True},
        {"turn_count": 2, "emotion": "sad", "success": True},
        {"turn_count": 3, "emotion": "joyful", "success": True},
    ]
    annotations = [
        {"turn_count": 1, "expected": "anxious"},
        {"turn_count": 2, "expected": "joyful"},
        {"turn_count": 3, "expected": "joyful"},
    ]

    result = evaluate_records(analysis_records, annotations)

    assert result["total"] == 3
    assert result["correct"] == 2
    assert round(result["accuracy"], 4) == 0.6667
    assert 0.20 < result["accuracy_ci95_low"] < result["accuracy"]
    assert result["accuracy"] < result["accuracy_ci95_high"] < 1.0
    assert round(result["macro_f1"], 4) == 0.5556
    assert round(result["family_accuracy"], 4) == 0.6667
    assert result["errors"] == [
        {
            "position": 1,
            "case_id": "",
            "turn_count": 2,
            "timestamp": None,
            "expected": "joyful",
            "predicted": "sad",
            "matched": True,
            "expected_family": "joy_contentment",
            "predicted_family": "sadness_loss",
            "family_match": False,
        }
    ]


def test_family_metric_marks_adjacent_intensity_error_without_hiding_exact_error():
    result = evaluate_records(
        [{"case_id": "fear", "emotion": "terrified", "success": True}],
        [{"case_id": "fear", "expected": "afraid"}],
    )

    assert result["accuracy"] == 0.0
    assert result["family_accuracy"] == 1.0
    assert result["errors"][0]["family_match"] is True


def test_evaluate_records_falls_back_to_successful_record_order():
    analysis_records = [
        {"turn_count": 1, "emotion": "", "success": False},
        {"turn_count": 2, "emotion": "grateful", "success": True},
    ]
    annotations = [{"expected": "grateful"}]

    result = evaluate_records(analysis_records, annotations)

    assert result["total"] == 1
    assert result["correct"] == 1
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_evaluate_records_matches_by_case_id_before_turn_count():
    analysis_records = [
        {"case_id": "case-002", "turn_count": 1, "emotion": "sad", "success": True},
        {"case_id": "case-001", "turn_count": 1, "emotion": "anxious", "success": True},
    ]
    annotations = [
        {"id": "case-001", "turn_count": 1, "expected": "anxious"},
        {"case_id": "case-002", "turn_count": 1, "expected": "sad"},
    ]

    result = evaluate_records(analysis_records, annotations)

    assert result["total"] == 2
    assert result["correct"] == 2
    assert result["accuracy"] == 1.0
    assert result["errors"] == []


def test_direct_cli_help_works():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_emotion_analysis.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Evaluate chatbot emotion-analysis records" in result.stdout
