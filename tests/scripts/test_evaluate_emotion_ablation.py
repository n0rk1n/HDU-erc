import json
import subprocess
import sys

from scripts.ablation.evaluate_emotion_ablation import compare_runs
from scripts.ablation.evaluate_emotion_ablation import main


def test_compare_runs_returns_metrics_per_run():
    annotations = [
        {"turn_count": 1, "expected": "anxious"},
        {"turn_count": 2, "expected": "grateful"},
    ]
    runs = {
        "baseline": [
            {"turn_count": 1, "emotion": "sad", "success": True},
            {"turn_count": 2, "emotion": "grateful", "success": True},
        ],
        "dynamic-eicl": [
            {"turn_count": 1, "emotion": "anxious", "success": True},
            {"turn_count": 2, "emotion": "grateful", "success": True},
        ],
    }

    result = compare_runs(runs, annotations)

    assert result["baseline"]["accuracy"] == 0.5
    assert result["dynamic-eicl"]["accuracy"] == 1.0
    assert "macro_f1" in result["dynamic-eicl"]


def test_main_creates_nested_output_directories(tmp_path):
    labels_file = tmp_path / "labels.json"
    run_file = tmp_path / "run.json"
    markdown_file = tmp_path / "reports" / "ablation" / "summary.md"
    csv_file = tmp_path / "reports" / "ablation" / "metrics.csv"
    labels_file.write_text(json.dumps([{"turn_count": 1, "expected": "anxious"}]), encoding="utf-8")
    run_file.write_text(
        json.dumps([{"turn_count": 1, "emotion": "anxious", "success": True}]),
        encoding="utf-8",
    )

    result = main([
        "--labels-file",
        str(labels_file),
        "--run",
        f"dynamic-eicl={run_file}",
        "--markdown-file",
        str(markdown_file),
        "--csv-file",
        str(csv_file),
    ])

    assert result == 0
    assert markdown_file.exists()
    assert csv_file.exists()


def test_main_returns_one_when_all_runs_have_zero_samples(tmp_path):
    labels_file = tmp_path / "labels.json"
    run_file = tmp_path / "run.json"
    markdown_file = tmp_path / "summary.md"
    csv_file = tmp_path / "metrics.csv"
    labels_file.write_text("[]", encoding="utf-8")
    run_file.write_text(
        json.dumps([{"turn_count": 1, "emotion": "anxious", "success": True}]),
        encoding="utf-8",
    )

    result = main([
        "--labels-file",
        str(labels_file),
        "--run",
        f"dynamic-eicl={run_file}",
        "--markdown-file",
        str(markdown_file),
        "--csv-file",
        str(csv_file),
    ])

    assert result == 1


def test_direct_cli_help_works():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ablation.evaluate_emotion_ablation",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Compare emotion ablation runs" in result.stdout
