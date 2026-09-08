"""Reproducible prompt-development evidence; reads only official training data."""

import csv
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import re

from emotion_lab.storage import Store, digest


def audit(source):
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("train.tsv", "emotions.txt"):
        assert digest((source / name).read_bytes()) == manifest["files"][name]["sha256"]
    labels = (source / "emotions.txt").read_text().splitlines()
    rows = [
        {"source_id": r[2], "text": r[0], "labels": [labels[int(i)] for i in r[1].split(",")]}
        for r in csv.reader((source / "train.tsv").open(), delimiter="\t", quoting=csv.QUOTE_NONE)
    ]
    counts = Counter(label for row in rows for label in row["labels"])
    pairs = Counter(pair for row in rows for pair in combinations(sorted(row["labels"]), 2))
    patterns = {
        "question_mark": r"\?",
        "profanity": r"\b(?:fuck\w*|shit\w*|damn\w*)\b",
        "negative_judgment_words": r"\b(?:wrong|stupid|dumb|bad|terrible)\b",
        "laughter": r"\b(?:lol|lmao|haha\w*)\b",
        "hope": r"\bhope\w*\b",
        "thanks": r"\b(?:thanks|thank you)\b",
        "dont_know": r"\bdon['’]t know\b",
    }
    rng = random.Random(908)
    strata = {
        label: rng.sample([r for r in rows if label in r["labels"]], 8)
        for label in labels
    }
    slices = {}
    for key, pattern in patterns.items():
        matched = [r for r in rows if re.search(pattern, r["text"], re.I)]
        neutral = [r for r in matched if r["labels"] == ["neutral"]]
        emotional = [r for r in matched if "neutral" not in r["labels"]]
        slices[key] = {
            "pattern": pattern,
            "samples": len(matched),
            "label_support": dict(Counter(l for r in matched for l in r["labels"])),
            "neutral_only": len(neutral),
            "neutral_examples": rng.sample(neutral, min(4, len(neutral))),
            "emotional_examples": rng.sample(emotional, min(4, len(emotional))),
        }
    return {
        "version": "train-prompt-audit-v3",
        "source_manifest": manifest,
        "input_files": {n: digest((source / n).read_bytes()) for n in ("train.tsv", "emotions.txt")},
        "selection": "Random(908), 8 rows per label in official order; 4 neutral and 4 non-neutral rows per regex slice",
        "sample_count": len(rows),
        "cardinality": dict(Counter(len(r["labels"]) for r in rows)),
        "neutral_only": sum(r["labels"] == ["neutral"] for r in rows),
        "neutral_cooccurrence": sum("neutral" in r["labels"] and len(r["labels"]) > 1 for r in rows),
        "label_support": dict(counts),
        "pair_support": {"/".join(k): v for k, v in sorted(pairs.items())},
        "label_strata": strata,
        "lexical_slices": slices,
        "scope": "Training-only descriptive evidence; regexes are audit slices, not classification rules. Previous dev diagnostics motivated this iteration; no claim of an untouched dev set.",
    }


if __name__ == "__main__":
    result = audit(Path("data/benchmarks/goemotions"))
    target = Path("data/research/train_prompt_v3_audit.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with Store("data/research") as store, store.transaction():
        aid = store.put({"analysis": result, "script": Path(__file__).read_text()}, "train_prompt_v3_audit", inline=False)
        store.event("artifact_id", aid, "train_prompt_v3_audited")
    print(json.dumps({"artifact_id": aid, "file": str(target), "samples": result["sample_count"], "cardinality": result["cardinality"], "neutral_only": result["neutral_only"]}))
