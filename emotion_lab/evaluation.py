"""Frozen multilabel scoring. No examples or model calls are made here."""

import csv
from pathlib import Path
import random

from .storage import digest, json_bytes, now

SCORER_VERSION = "native-multilabel-v1"


def ratio(n, d, zero):
    return n / d if d else float(zero)


def _metric(
    store, eid, name, scope, value, n=None, d=None, support=None, undefined=None
):
    store.insert(
        "metric_values",
        evaluation_id=eid,
        metric_name=name,
        scope_key=scope,
        value=value,
        numerator=n,
        denominator=d,
        support=support,
        undefined_reason=undefined,
    )


def evaluate(store, run_id, config):
    config = dict(config)
    if set(config) - {"zero_division", "scope", "parser_version"}:
        raise ValueError("unsupported scoring configuration")
    zero = config.setdefault("zero_division", 0)
    if type(zero) is not int or zero not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")
    config.update(
        label_universe="native-28",
        failure_policy="empty-prediction",
        scorer_version=SCORER_VERSION,
    )
    run = store.one(
        "SELECT r.*,e.dataset_version_id,e.evaluation_set_id FROM experiment_runs r JOIN experiments e USING(experiment_id) WHERE run_id=?",
        (run_id,),
    )
    sample_set = store.one(
        "SELECT * FROM sample_sets WHERE sample_set_id=?", (run["evaluation_set_id"],)
    )
    items = store.rows(
        "SELECT i.*,s.source_id,s.raw_text FROM run_items i JOIN samples s USING(sample_id) WHERE run_id=? ORDER BY ordinal",
        (run_id,),
    )
    members = store.rows(
        "SELECT sample_id FROM sample_set_members WHERE sample_set_id=? ORDER BY ordinal",
        (sample_set["sample_set_id"],),
    )
    if [i["sample_id"] for i in items] != [m["sample_id"] for m in members]:
        raise ValueError("planned samples differ from frozen set")
    labels = store.rows(
        "SELECT * FROM label_definitions WHERE dataset_version_id=? ORDER BY label_id",
        (run["dataset_version_id"],),
    )
    names = {l["label_id"]: l["label_name"] for l in labels}
    if len(names) != 28:
        raise ValueError("expected 28 native labels")
    truth = {i["sample_id"]: set() for i in items}
    for row in store.rows(
        "SELECT l.sample_id,l.label_id FROM sample_labels l JOIN run_items i USING(sample_id) WHERE i.run_id=?",
        (run_id,),
    ):
        truth[row["sample_id"]].add(row["label_id"])
    chosen = {}
    predicted = {}
    snapshot = []
    for item in items:
        pid = item["final_prediction_id"] if item["status"] == "succeeded" else None
        if config.get("parser_version"):
            # Re-scoring parser versions uses deterministic attempt order, never truth labels.
            rows = store.rows(
                "SELECT p.prediction_id FROM predictions p JOIN call_attempts a USING(call_attempt_id) WHERE p.run_item_id=? AND p.parser_version=? AND p.status='valid' ORDER BY a.attempt_no LIMIT 1",
                (item["run_item_id"], config["parser_version"]),
            )
            pid = rows[0]["prediction_id"] if rows else None
        chosen[item["run_item_id"]] = pid
        predicted[item["run_item_id"]] = (
            {
                r["label_id"]
                for r in store.rows(
                    "SELECT label_id FROM prediction_labels WHERE prediction_id=?",
                    (pid,),
                )
            }
            if pid
            else set()
        )
        if pid:
            p = store.one(
                "SELECT p.*,a.response_artifact_id FROM predictions p JOIN call_attempts a USING(call_attempt_id) WHERE prediction_id=?",
                (pid,),
            )
            store.read(p["response_artifact_id"])
            store.read(p["validation_artifact_id"])
        snapshot.append(
            {
                "run_item_id": item["run_item_id"],
                "sample_id": item["sample_id"],
                "prediction_id": pid,
                "status_at_evaluation": item["status"],
                "missing_reason": None
                if pid
                else item["failure_code"] or item["status"],
            }
        )
    observed = sum(i["status"] in ("succeeded", "failed") for i in items)
    failed = sum(i["status"] == "failed" for i in items)
    split_count = store.one(
        "SELECT COUNT(*) n FROM samples WHERE dataset_version_id=? AND split=?",
        (run["dataset_version_id"], sample_set["source_split"]),
    )["n"]
    default_scope = (
        "diagnostic_partial"
        if observed != len(items)
        else ("official_full" if len(items) == split_count else "diagnostic_slice")
    )
    scope = config.get("scope", default_scope)
    if scope == "official_full" and (
        observed != len(items) or len(items) != split_count
    ):
        raise ValueError("official_full requires the entire completed official split")
    if scope not in {"official_full", "diagnostic_partial", "diagnostic_slice"}:
        raise ValueError("invalid scoring scope")
    if observed != len(items) and scope != "diagnostic_partial":
        raise ValueError("incomplete run requires diagnostic_partial")
    config["scope"] = scope
    family = store.json(run["taxonomy_config_artifact_id"])["families"]
    counts = {i: [0, 0, 0] for i in names}
    family_counts = {f: [0, 0, 0] for f in set(family.values())}
    totals = [0, 0, 0]
    exact = 0
    with store.transaction():
        eid = store.insert(
            "evaluations",
            run_id=run_id,
            sample_set_id=sample_set["sample_set_id"],
            scorer_version=SCORER_VERSION,
            evaluation_config_artifact_id=store.put(config, "evaluation_config"),
            prediction_snapshot_artifact_id=store.put(snapshot, "prediction_snapshot"),
            ground_truth_manifest_sha256=store.one(
                "SELECT label_manifest_sha256 FROM dataset_versions WHERE dataset_version_id=?",
                (run["dataset_version_id"],),
            )["label_manifest_sha256"],
            family_mapping_artifact_id=store.put(family, "family_mapping"),
            scope=scope,
            status="building",
            expected_items=len(items),
            observed_items=observed,
            failed_items=failed,
        )
        for item, snap in zip(items, snapshot):
            actual = truth[item["sample_id"]]
            pred = predicted[item["run_item_id"]]
            tp, fp, fn = len(actual & pred), len(pred - actual), len(actual - pred)
            match = int(actual == pred)
            exact += match
            totals = [totals[0] + tp, totals[1] + fp, totals[2] + fn]
            for label in actual | pred:
                counts[label][0] += int(label in actual and label in pred)
                counts[label][1] += int(label in pred and label not in actual)
                counts[label][2] += int(label in actual and label not in pred)
            af = {family[names[i]] for i in actual}
            pf = {family[names[i]] for i in pred}
            for f in af | pf:
                family_counts[f][0] += int(f in af and f in pf)
                family_counts[f][1] += int(f in pf and f not in af)
                family_counts[f][2] += int(f in af and f not in pf)
            outcome = (
                ("correct" if match else "incorrect")
                if snap["prediction_id"]
                else ("failed" if item["status"] == "failed" else "missing")
            )
            store.insert(
                "evaluation_items",
                evaluation_id=eid,
                run_item_id=item["run_item_id"],
                prediction_id=snap["prediction_id"],
                outcome=outcome,
                missing_reason=snap["missing_reason"],
                tp=tp,
                fp=fp,
                fn=fn,
                exact_match=match,
                missing_label_ids_json=sorted(actual - pred),
                extra_label_ids_json=sorted(pred - actual),
                slice_memberships_json={
                    "multilabel": len(actual) > 1,
                    "neutral_cooccurrence": 27 in actual and len(actual) > 1,
                },
            )
        macro = []
        for scope_prefix, group in [("label", counts), ("family", family_counts)]:
            for label, (tp, fp, fn) in group.items():
                key = (
                    scope_prefix
                    + "/"
                    + (names[label] if scope_prefix == "label" else label)
                )
                f1 = ratio(2 * tp, 2 * tp + fp + fn, zero)
                if scope_prefix == "label":
                    macro.append(f1)
                for metric, n, d in [
                    ("precision", tp, tp + fp),
                    ("recall", tp, tp + fn),
                    ("f1", 2 * tp, 2 * tp + fp + fn),
                ]:
                    _metric(
                        store,
                        eid,
                        metric,
                        key,
                        ratio(n, d, zero),
                        n,
                        d,
                        tp + fn,
                        "zero_denominator" if not d else None,
                    )
        tp, fp, fn = totals
        _metric(
            store,
            eid,
            "micro_f1",
            "overall",
            ratio(2 * tp, 2 * tp + fp + fn, zero),
            2 * tp,
            2 * tp + fp + fn,
            len(items),
        )
        _metric(
            store,
            eid,
            "macro_f1",
            "overall",
            sum(macro) / len(names),
            sum(macro),
            len(names),
            len(items),
        )
        _metric(
            store,
            eid,
            "exact_match",
            "overall",
            exact / len(items),
            exact,
            len(items),
            len(items),
        )
        _metric(
            store,
            eid,
            "failure_rate",
            "overall",
            failed / len(items),
            failed,
            len(items),
            len(items),
        )
        missing = sum(s["prediction_id"] is None for s in snapshot)
        _metric(
            store,
            eid,
            "missing_prediction_rate",
            "overall",
            missing / len(items),
            missing,
            len(items),
            len(items),
        )
        costrows = store.rows(
            "SELECT currency,SUM(estimated_cost_micros) cost FROM v_call_audit WHERE run_id=? AND estimated_cost_micros IS NOT NULL GROUP BY currency",
            (run_id,),
        )
        for row in costrows:
            _metric(
                store,
                eid,
                "known_estimated_cost_micros",
                "currency/" + row["currency"],
                row["cost"],
            )
        unknown = store.one(
            "SELECT COUNT(*) n FROM v_call_audit WHERE run_id=? AND estimated_cost_micros IS NULL AND status!='cancelled_before_dispatch'",
            (run_id,),
        )["n"]
        _metric(store, eid, "unknown_cost_calls", "overall", unknown)
        store.transition(
            "evaluations", "evaluation_id", eid, "completed", completed_at=now()
        )
    return eid


def compare(store, baseline, candidate, seed=42, repeats=1000, confidence=0.95):
    if type(repeats) is not int or repeats < 20 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    left = store.one("SELECT * FROM evaluations WHERE evaluation_id=?", (baseline,))
    right = store.one("SELECT * FROM evaluations WHERE evaluation_id=?", (candidate,))
    if left["status"] != "completed" or right["status"] != "completed":
        raise ValueError("comparisons require frozen evaluations")
    lc = store.json(left["evaluation_config_artifact_id"])
    rc = store.json(right["evaluation_config_artifact_id"])
    if (
        lc != rc
        or left["ground_truth_manifest_sha256"] != right["ground_truth_manifest_sha256"]
    ):
        raise ValueError("scoring definitions differ")
    ls = store.one(
        "SELECT dataset_version_id FROM sample_sets WHERE sample_set_id=?",
        (left["sample_set_id"],),
    )
    rs = store.one(
        "SELECT dataset_version_id FROM sample_sets WHERE sample_set_id=?",
        (right["sample_set_id"],),
    )
    if ls != rs:
        raise ValueError("dataset versions differ")

    def rows(eid):
        return {
            r["sample_id"]: r
            for r in store.rows(
                "SELECT i.sample_id,e.tp,e.fp,e.fn FROM evaluation_items e JOIN run_items i USING(run_item_id) WHERE evaluation_id=?",
                (eid,),
            )
        }

    a, b = rows(baseline), rows(candidate)
    if set(a) != set(b):
        raise ValueError("paired sample IDs differ; no implicit intersection")
    ids = sorted(a)
    zero = lc["zero_division"]

    def f1(data, selected):
        tp = sum(data[i]["tp"] for i in selected)
        fp = sum(data[i]["fp"] for i in selected)
        fn = sum(data[i]["fn"] for i in selected)
        return ratio(2 * tp, 2 * tp + fp + fn, zero)

    rng = random.Random(seed)
    diffs = []
    for _ in range(repeats):
        selected = rng.choices(ids, k=len(ids))
        diffs.append(f1(b, selected) - f1(a, selected))
    diffs.sort()
    alpha = (1 - confidence) / 2
    low = diffs[int(alpha * (repeats - 1))]
    high = diffs[int((1 - alpha) * (repeats - 1))]
    result = {
        "micro_f1_delta": f1(b, ids) - f1(a, ids),
        "ci_low": low,
        "ci_high": high,
        "samples": len(ids),
        "bootstrap_deltas": diffs,
    }
    with store.transaction():
        cid = store.insert(
            "comparisons",
            baseline_evaluation_id=baseline,
            candidate_evaluation_id=candidate,
            analysis_config_artifact_id=store.put(
                {
                    "version": "paired-percentile-v1",
                    "seed": seed,
                    "repeats": repeats,
                    "confidence": confidence,
                    "direction": "candidate-minus-baseline",
                },
                "comparison_config",
            ),
            paired_members_artifact_id=store.put(ids, "paired_members"),
            result_artifact_id=store.put(result, "comparison_result"),
            status="building",
        )
        store.insert(
            "metric_values",
            comparison_id=cid,
            metric_name="micro_f1_delta",
            scope_key="overall",
            value=result["micro_f1_delta"],
            support=len(ids),
            ci_low=low,
            ci_high=high,
        )
        store.transition(
            "comparisons", "comparison_id", cid, "completed", completed_at=now()
        )
    return cid


def export_evaluation(store, evaluation_id, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("export output exists")
    evaluation = store.one(
        "SELECT * FROM evaluations WHERE evaluation_id=?", (evaluation_id,)
    )
    if evaluation["status"] != "completed":
        raise ValueError("export requires frozen evaluation")
    output.mkdir(parents=True)
    rows = store.rows(
        "SELECT e.*,i.sample_id,s.source_id,s.raw_text FROM evaluation_items e JOIN run_items i USING(run_item_id) JOIN samples s USING(sample_id) WHERE evaluation_id=? ORDER BY i.ordinal",
        (evaluation_id,),
    )
    for row in rows:
        row["true_labels"] = [
            r["label_id"]
            for r in store.rows(
                "SELECT label_id FROM sample_labels WHERE sample_id=? ORDER BY label_id",
                (row["sample_id"],),
            )
        ]
        row["predicted_labels"] = [
            r["label_id"]
            for r in store.rows(
                "SELECT label_id FROM prediction_labels WHERE prediction_id=? ORDER BY label_id",
                (row["prediction_id"],),
            )
        ]
    (output / "predictions.jsonl").write_bytes(
        b"".join(json_bytes(r) + b"\n" for r in rows)
    )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = store.rows(
        "SELECT * FROM metric_values WHERE evaluation_id=?", (evaluation_id,)
    )
    (output / "metrics.json").write_bytes(json_bytes(metrics))
    run = store.one(
        "SELECT * FROM experiment_runs WHERE run_id=?", (evaluation["run_id"],)
    )
    (output / "configuration.json").write_bytes(
        json_bytes(
            {
                "run": run,
                "evaluation": evaluation,
                "run_config": store.json(run["method_config_artifact_id"]),
                "scoring": store.json(evaluation["evaluation_config_artifact_id"]),
                "snapshot": store.json(evaluation["prediction_snapshot_artifact_id"]),
            }
        )
    )
    # Evidence manifest refers to immutable database IDs; use storage backup for portable full evidence.
    artifacts = store.rows(
        "SELECT artifact_id,kind,content_sha256,storage_sha256,relative_path,logical_bytes FROM artifacts"
    )
    (output / "evidence_manifest.json").write_bytes(json_bytes(artifacts))
    files = {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    manifest = {
        "evaluation_id": evaluation_id,
        "run_id": evaluation["run_id"],
        "exporter": "emotion-lab-v1",
        "sample_count": len(rows),
        "files": files,
        "created_at": now(),
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    with store.transaction():
        aid = store.put(manifest, "export_manifest")
        store.event("evaluation_id", evaluation_id, "exported", payload=aid)
    return output
