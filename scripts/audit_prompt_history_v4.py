"""Archive all prior dev cases and fresh train-only boundary evidence for v4."""

import csv
import json
from pathlib import Path
import random

from emotion_lab.storage import Store, digest


def main():
    root = Path("data/research")
    earlier = json.loads((root / "glm_prompt_comparison_analysis.json").read_text())
    recent = json.loads((root / "train_prompt_v3_analysis.json").read_text())
    first = {r["source_id"]: r for r in map(json.loads, (root / "exports/glm-dev200-paired-cases-20260908.jsonl").read_text().splitlines())}
    last = {r["source_id"]: r for r in map(json.loads, (root / "exports/glm-dev200-prompt-v3-paired-cases-20260908.jsonl").read_text().splitlines())}
    assert set(first) == set(last) and len(first) == 200
    families = json.loads(Path("config/emotion_families.json").read_text())
    history = []
    for sid in sorted(first):
        a, b = first[sid], last[sid]
        assert a["text"] == b["text"] and a["truth"] == b["truth"]
        truth = set(a["truth"])
        observations = {}
        for name, row in [("v1_ranked", a["baseline"]), ("v2_first", a["revised"]), ("v2_repeat", b["control"]), ("v3", b["candidate"])]:
            pred = set(row["predicted_labels"])
            observations[name] = {
                "prediction_id": row["prediction_id"], "labels": sorted(pred),
                "missing": sorted(truth - pred), "extra": sorted(pred - truth),
                "exact": truth == pred, "any_hit": bool(truth & pred),
                "family_any_hit": bool({families[x] for x in truth} & {families[x] for x in pred}),
                "selected_examples": row["selected_examples"],
            }
        history.append({"source_id": sid, "text": a["text"], "truth": a["truth"], "observations": observations})
    source = Path("data/benchmarks/goemotions")
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("train.tsv", "emotions.txt"):
        assert digest((source / name).read_bytes()) == manifest["files"][name]["sha256"]
    labels = (source / "emotions.txt").read_text().splitlines()
    with (source / "train.tsv").open() as stream:
        train = [{"source_id": r[2], "text": r[0], "labels": [labels[int(i)] for i in r[1].split(",")]} for r in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)]
    rng = random.Random(904)
    targets = ["disappointment", "realization", "approval", "caring", "confusion", "remorse"]
    reviewed = {label: rng.sample([r for r in train if label in r["labels"]], 8) for label in targets}
    total_gold = sum(len(r["truth"]) for r in history)
    neutral_tp = sum("neutral" in r["truth"] for r in history)
    summary = {
        "historical_metrics": {"v1_ranked": earlier["metrics"]["baseline"], "v2_first": earlier["metrics"]["revised"], "v2_repeat": recent["metrics"]["control"], "v3": recent["metrics"]["candidate"]},
        "historical_analysis_artifacts": [earlier["analysis_artifact_id"], recent["analysis_artifact_id"]],
        "prior_train_audit_artifact_id": "df0b445f-fe1b-4bea-bcb1-961f99778c41",
        "source_hashes": {name: digest((source / name).read_bytes()) for name in ("train.tsv", "emotions.txt")},
        "train_boundary_samples": reviewed,
        "train_sampling": "Random(904); 8 examples per target label in declared order; original text and full annotations",
        "always_neutral_reference": {"kind": "deterministic offline reference, not a model run", "n": 200, "exact_match": sum(r["truth"] == ["neutral"] for r in history) / 200, "micro_f1": 2 * neutral_tp / (200 + total_gold)},
        "stable_no_overlap_all_four": sum(all(not p["any_hit"] for p in r["observations"].values()) for r in history),
        "v2_repeat_label_sets_changed": sum(r["observations"]["v2_first"]["labels"] != r["observations"]["v2_repeat"]["labels"] for r in history),
        "scope": "Prior dev errors deliberately inform v4 hypotheses; this is development, not train-only tuning or independent validation. No dev text/label pairs are inserted into the new prompt; official test is unused.",
    }
    with Store(root) as store, store.transaction():
        history_id = store.put({"cases": history}, "prompt_v4_prior_case_history", inline=False)
        summary["prior_case_history_artifact_id"] = history_id
        aid = store.put({"analysis": summary, "script": Path(__file__).read_text()}, "prompt_v4_development_basis", inline=False)
        store.event("artifact_id", aid, "prompt_v4_basis_frozen", payload=history_id)
    summary["artifact_id"] = aid
    (root / "prompt_v4_development_basis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["artifact_id", "prior_case_history_artifact_id", "always_neutral_reference", "stable_no_overlap_all_four", "v2_repeat_label_sets_changed"]}))


if __name__ == "__main__":
    main()
