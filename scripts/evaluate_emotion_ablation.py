"""Compare emotion-recognition ablation runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from scripts.evaluate_emotion_analysis import evaluate_records, load_records


def compare_runs(
    runs: dict[str, list[dict[str, Any]]],
    annotations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: evaluate_records(records, annotations)
        for name, records in runs.items()
    }


def markdown_table(results: dict[str, dict[str, Any]]) -> str:
    lines = ["| Run | Samples | Accuracy | Macro F1 |", "| --- | ---: | ---: | ---: |"]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['total']} | {result['accuracy'] * 100:.2f}% | "
            f"{result['macro_f1'] * 100:.2f}% |"
        )
    return "\n".join(lines)


def write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run", "samples", "correct", "accuracy", "macro_f1"])
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({
                "run": name,
                "samples": result["total"],
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
            })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare emotion ablation runs.")
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=path/to/analysis.json")
    parser.add_argument("--markdown-file", required=True)
    parser.add_argument("--csv-file", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    annotations = load_records(Path(args.labels_file))
    runs = {}
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must use NAME=PATH")
        name, path = item.split("=", 1)
        runs[name] = load_records(Path(path))
    results = compare_runs(runs, annotations)
    markdown = markdown_table(results)
    Path(args.markdown_file).write_text(markdown + "\n", encoding="utf-8")
    write_csv(Path(args.csv_file), results)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
