"""Export emotion ablation v2 records to the legacy ablation JSONL format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmark.emotion_benchmark import export_dialogue
from scripts.benchmark.emotion_benchmark import export_label
from scripts.benchmark.emotion_benchmark import load_jsonl
from scripts.benchmark.emotion_benchmark import validate_records
from scripts.benchmark.emotion_benchmark import write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export emotion ablation v2 records.")
    parser.add_argument("--input", required=True, help="Path to benchmark JSONL records.")
    parser.add_argument("--output-dir", required=True, help="Directory for dialogues.jsonl and labels.jsonl.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_jsonl(Path(args.input))
    errors = validate_records(records)
    if errors:
        for error in errors:
            print(error)
        return 1

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "dialogues.jsonl", [export_dialogue(record) for record in records])
    write_jsonl(output_dir / "labels.jsonl", [export_label(record) for record in records])
    print(f"Exported {len(records)} records to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
