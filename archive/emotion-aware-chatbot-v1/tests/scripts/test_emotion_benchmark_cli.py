import json
import subprocess
import sys


PUBLIC_SEED = "data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl"


def test_validate_cli_accepts_public_seed_release():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/validate_emotion_benchmark.py",
            "--input",
            PUBLIC_SEED,
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
            "subset": "empathetic_dialogues_test",
            "expected": "worried",
            "turn_count": 1,
            "history": [],
            "current_input": "I cannot stop thinking about it.",
            "scenario": "open_domain_dialogue",
            "annotation_status": "released",
            "label_provenance": "human_authored_emotion_grounding",
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


def test_export_cli_writes_public_dialogues_and_labels(tmp_path):
    output_dir = tmp_path / "exported"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/export_emotion_benchmark.py",
            "--input",
            PUBLIC_SEED,
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
    assert set(labels[0]) == {"id", "expected", "label_provenance"}


def test_summary_cli_writes_public_distribution_csvs(tmp_path):
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/summarize_emotion_benchmark.py",
            "--input",
            PUBLIC_SEED,
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
