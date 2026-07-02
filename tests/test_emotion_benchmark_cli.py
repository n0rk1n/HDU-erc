import json
import subprocess
import sys

import pytest


def test_validate_cli_accepts_seed_release():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/validate_emotion_benchmark.py",
            "--input",
            "data/benchmarks/emotion_ablation_v2/release/seed.jsonl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Validated 64 records" in result.stdout


def test_validate_cli_rejects_invalid_record(tmp_path):
    input_file = tmp_path / "bad.jsonl"
    input_file.write_text(
        json.dumps({
            "case_id": "bad-1",
            "language": "en",
            "subset": "seed",
            "expected": "worried",
            "turn_count": 1,
            "history": [],
            "current_input": "I cannot stop thinking about it.",
            "scenario": "workplace_interview",
            "annotation_status": "released",
        })
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/validate_emotion_benchmark.py",
            "--input",
            str(input_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "expected must be one of the supported emotion labels" in result.stdout


def test_export_cli_writes_dialogues_and_labels(tmp_path):
    output_dir = tmp_path / "exported"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/export_emotion_ablation_v2.py",
            "--input",
            "data/benchmarks/emotion_ablation_v2/release/seed.jsonl",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Exported 64 records" in result.stdout

    dialogues = [
        json.loads(line)
        for line in (output_dir / "dialogues.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [
        json.loads(line)
        for line in (output_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(dialogues) == 64
    assert len(labels) == 64
    assert set(dialogues[0]) == {"id", "turn_count", "history", "current_input", "notes"}
    assert set(labels[0]) == {"id", "expected"}


def test_summary_cli_writes_distribution_csvs(tmp_path):
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/summarize_emotion_benchmark.py",
            "--input",
            "data/benchmarks/emotion_ablation_v2/release/seed.jsonl",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Summarized 64 records" in result.stdout
    assert (output_dir / "label_distribution.csv").exists()
    assert (output_dir / "scenario_distribution.csv").exists()
    assert "label,count" in (output_dir / "label_distribution.csv").read_text(encoding="utf-8")


def test_parallel_check_cli_accepts_seed_pairs():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/check_parallel_equivalence.py",
            "--input",
            "data/benchmarks/emotion_ablation_v2/release/seed.jsonl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Parallel check passed" in result.stdout


def test_parallel_check_cli_rejects_mismatched_pair(tmp_path):
    input_file = tmp_path / "pairs.jsonl"
    records = [
        {
            "case_id": "pair-1-en",
            "pair_id": "pair-1",
            "language": "en",
            "subset": "core_parallel",
            "expected": "anxious",
            "intensity": 0.8,
            "turn_count": 1,
            "history": [],
            "current_input": "I keep checking the result page.",
            "scenario": "academic_exam",
            "annotation_status": "released",
        },
        {
            "case_id": "pair-1-zh",
            "pair_id": "pair-1",
            "language": "zh",
            "subset": "core_parallel",
            "expected": "joyful",
            "intensity": 0.8,
            "turn_count": 1,
            "history": [],
            "current_input": "我一直刷新成绩页面。",
            "scenario": "academic_exam",
            "annotation_status": "released",
        },
    ]
    input_file.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/check_parallel_equivalence.py",
            "--input",
            str(input_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "pair-1: expected labels differ" in result.stdout


@pytest.mark.parametrize("max_intensity_delta", ["nan", "inf", "-0.1"])
def test_parallel_check_cli_rejects_invalid_intensity_delta(max_intensity_delta):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/check_parallel_equivalence.py",
            "--input",
            "data/benchmarks/emotion_ablation_v2/release/seed.jsonl",
            "--max-intensity-delta",
            max_intensity_delta,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "--max-intensity-delta" in result.stderr
