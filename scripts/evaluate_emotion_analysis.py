"""Evaluate stored emotion-analysis records against manual labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.emotion import EMOTION_ANALYSIS_FILE, EMOTION_LABEL_SET


def load_records(path: Path) -> list[dict[str, Any]]:
    data = _load_json_or_jsonl(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array or JSONL objects.")
    return [item for item in data if isinstance(item, dict)]


def evaluate_records(
    analysis_records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = _match_pairs(analysis_records, annotations)
    labels = sorted({
        label
        for pair in pairs
        for label in (pair["expected"], pair["predicted"])
        if label
    })
    correct = sum(1 for pair in pairs if pair["expected"] == pair["predicted"])
    total = len(pairs)
    accuracy = correct / total if total else 0.0
    macro_f1 = _macro_f1(pairs, labels) if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": labels,
        "errors": [
            pair for pair in pairs
            if pair["expected"] != pair["predicted"]
        ],
    }


def _load_json_or_jsonl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        return records


def _match_pairs(
    analysis_records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    successful_records = [
        record for record in analysis_records
        if record.get("success") is True and _normalize_label(record.get("emotion"))
    ]
    pairs = []

    for position, annotation in enumerate(annotations):
        expected = _expected_label(annotation)
        if expected is None:
            continue

        record = _find_record(annotation, analysis_records, successful_records, position)
        predicted = _normalize_label(record.get("emotion")) if record else ""
        pairs.append({
            "position": position,
            "turn_count": annotation.get("turn_count") if record is None else record.get("turn_count"),
            "timestamp": annotation.get("timestamp") if record is None else record.get("timestamp"),
            "expected": expected,
            "predicted": predicted,
            "matched": record is not None,
        })

    return pairs


def _expected_label(annotation: dict[str, Any]) -> str | None:
    for key in ("expected", "emotion", "label"):
        label = _normalize_label(annotation.get(key))
        if label:
            return label
    return None


def _normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    label = value.strip().lower()
    if not label:
        return ""
    return label if label in EMOTION_LABEL_SET else label


def _find_record(
    annotation: dict[str, Any],
    analysis_records: list[dict[str, Any]],
    successful_records: list[dict[str, Any]],
    position: int,
) -> dict[str, Any] | None:
    index = annotation.get("index")
    if type(index) is int and 0 <= index < len(successful_records):
        return successful_records[index]

    turn_count = annotation.get("turn_count")
    if type(turn_count) is int:
        for record in analysis_records:
            if record.get("turn_count") == turn_count:
                return record

    timestamp = annotation.get("timestamp")
    if isinstance(timestamp, str):
        for record in analysis_records:
            if record.get("timestamp") == timestamp:
                return record

    if position < len(successful_records):
        return successful_records[position]
    return None


def _macro_f1(pairs: list[dict[str, Any]], labels: list[str]) -> float:
    expected_counts = Counter(pair["expected"] for pair in pairs)
    predicted_counts = Counter(pair["predicted"] for pair in pairs)
    true_positive = Counter(
        pair["expected"] for pair in pairs
        if pair["expected"] == pair["predicted"]
    )

    scores = []
    for label in labels:
        tp = true_positive[label]
        fp = predicted_counts[label] - tp
        fn = expected_counts[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def print_report(result: dict[str, Any], max_errors: int) -> None:
    print(f"Samples: {result['total']}")
    print(f"Correct: {result['correct']}")
    print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {result['macro_f1'] * 100:.2f}%")

    errors = result["errors"]
    if not errors:
        print("Errors: 0")
        return

    print(f"Errors: {len(errors)}")
    for error in errors[:max_errors]:
        marker = error.get("turn_count")
        if marker is None:
            marker = f"index={error['position']}"
        print(
            f"- {marker}: expected={error['expected']} "
            f"predicted={error['predicted'] or '<missing>'}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chatbot emotion-analysis records against manual labels.",
    )
    parser.add_argument(
        "--analysis-file",
        default=EMOTION_ANALYSIS_FILE,
        help="Path to emotion_analysis.json. Defaults to the chatbot data file.",
    )
    parser.add_argument(
        "--labels-file",
        required=True,
        help="JSON or JSONL file with expected labels.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum number of mismatches to print.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis_records = load_records(Path(args.analysis_file))
    annotations = load_records(Path(args.labels_file))
    result = evaluate_records(analysis_records, annotations)
    print_report(result, max(0, args.max_errors))
    return 0 if result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
