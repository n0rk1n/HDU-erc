"""Verify the pinned official files and report dataset structure; no model calls."""

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        content = (root / name).read_bytes()
        require(len(content) == expected["bytes"], f"Size mismatch: {name}")
        require(
            hashlib.sha256(content).hexdigest() == expected["sha256"],
            f"SHA-256 mismatch: {name}",
        )

    labels = (root / "emotions.txt").read_text(encoding="utf-8").splitlines()
    require(len(labels) == len(set(labels)) == 28, "Expected 28 unique labels")
    report = {
        "source_commit": manifest["source_commit"],
        "verified_source_files": len(manifest["files"]),
        "normalization": "casefold, trim and collapse whitespace; no punctuation removal",
        "splits": {},
        "cross_split_overlap": {},
    }
    split_data = {}
    for split, expected_rows in manifest["expected_rows"].items():
        ids, texts, normalized = [], [], []
        counts, cardinality = Counter(), Counter()
        neutral_with_emotion = 0
        lines = (root / f"{split}.tsv").read_text(encoding="utf-8").splitlines()
        require(len(lines) == expected_rows, f"Wrong row count: {split}")
        for number, line in enumerate(lines, 1):
            fields = line.split("\t")
            require(len(fields) == 3, f"Wrong TSV columns: {split}:{number}")
            text, label_text, comment_id = fields
            require(text.strip() and comment_id.strip(), f"Empty field: {split}:{number}")
            target = [int(value) for value in label_text.split(",")]
            require(
                len(target) == len(set(target)) and all(0 <= value < len(labels) for value in target),
                f"Invalid labels: {split}:{number}",
            )
            ids.append(comment_id)
            texts.append(text)
            normalized.append(" ".join(text.casefold().split()))
            counts.update(labels[value] for value in target)
            cardinality[len(target)] += 1
            neutral_with_emotion += int(labels.index("neutral") in target and len(target) > 1)
        require(len(ids) == len(set(ids)), f"Duplicate comment IDs within {split}")
        split_data[split] = {"id": ids, "exact_text": texts, "normalized_text": normalized}
        report["splits"][split] = {
            "rows": len(ids),
            "unique_ids": len(set(ids)),
            "unique_exact_texts": len(set(texts)),
            "unique_normalized_texts": len(set(normalized)),
            "multilabel_rows": sum(n for k, n in cardinality.items() if k > 1),
            "label_cardinality": dict(sorted(cardinality.items())),
            "neutral_with_emotion_rows": neutral_with_emotion,
            "label_positive_counts": {label: counts[label] for label in labels},
        }
    for left, right in combinations(split_data, 2):
        overlap = {}
        for field in ("id", "exact_text", "normalized_text"):
            shared = set(split_data[left][field]) & set(split_data[right][field])
            overlap[field] = {
                "shared_values": len(shared),
                f"{left}_rows": sum(value in shared for value in split_data[left][field]),
                f"{right}_rows": sum(value in shared for value in split_data[right][field]),
            }
        require(overlap["id"]["shared_values"] == 0, f"Cross-split ID overlap: {left}/{right}")
        report["cross_split_overlap"][f"{left}/{right}"] = overlap
    report["total_rows"] = sum(split["rows"] for split in report["splits"].values())
    return report


if __name__ == "__main__":
    print(json.dumps(audit(Path(__file__).resolve().parent), ensure_ascii=False, indent=2))
