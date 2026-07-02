"""Check bilingual parallel equivalence for emotion ablation v2 records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmark.emotion_benchmark import load_jsonl
from scripts.benchmark.emotion_benchmark import parallel_equivalence_errors
from scripts.benchmark.emotion_benchmark import validate_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check bilingual parallel benchmark pairs.")
    parser.add_argument("--input", required=True, help="Path to benchmark JSONL records.")
    parser.add_argument("--max-intensity-delta", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_jsonl(Path(args.input))
    errors = validate_records(records)
    errors.extend(
        parallel_equivalence_errors(
            records,
            max_intensity_delta=args.max_intensity_delta,
        )
    )
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Parallel check passed for {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
