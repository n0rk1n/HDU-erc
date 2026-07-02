"""Summarize emotion ablation v2 benchmark distributions."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmark.emotion_benchmark import load_jsonl
from scripts.benchmark.emotion_benchmark import summarize_records
from scripts.benchmark.emotion_benchmark import validate_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize emotion ablation v2 records.")
    parser.add_argument("--input", required=True, help="Path to benchmark JSONL records.")
    parser.add_argument("--output-dir", required=True, help="Directory for distribution CSV files.")
    return parser.parse_args(argv)


def write_distribution(path: Path, key_name: str, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([key_name, "count"])
        for key, count in sorted(counts.items()):
            if not key:
                continue
            writer.writerow([key, count])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_jsonl(Path(args.input))
    errors = validate_records(records)
    if errors:
        for error in errors:
            print(error)
        return 1

    output_dir = Path(args.output_dir)
    summary = summarize_records(records)
    write_distribution(output_dir / "label_distribution.csv", "label", summary["label"])
    write_distribution(output_dir / "language_distribution.csv", "language", summary["language"])
    write_distribution(output_dir / "subset_distribution.csv", "subset", summary["subset"])
    write_distribution(output_dir / "scenario_distribution.csv", "scenario", summary["scenario"])
    write_distribution(output_dir / "ambiguity_distribution.csv", "ambiguity", summary["ambiguity_level"])
    write_distribution(
        output_dir / "context_dependency_distribution.csv",
        "context_dependency",
        summary["context_dependency"],
    )
    write_distribution(output_dir / "quality_flag_distribution.csv", "quality_flag", summary["quality_flags"])
    print(f"Summarized {len(records)} records into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
