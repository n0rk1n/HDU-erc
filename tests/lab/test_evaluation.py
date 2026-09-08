import sqlite3
import pytest
from emotion_lab.runner import prepare_run, execute_run, reparse
from emotion_lab.evaluation import evaluate, compare, export_evaluation
from test_runner import Counter, response


def test_hand_calculated_multilabel_metrics_and_frozen_snapshot(configured, tmp_path):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())
    execute_run(store, run, transport=lambda request: response(), counter=Counter())
    eid = evaluate(store, run, {})
    values = {
        r["metric_name"]: r["value"]
        for r in store.rows(
            "SELECT * FROM metric_values WHERE evaluation_id=? AND scope_key='overall'",
            (eid,),
        )
    }
    # truth [{joy,neutral},{sadness}], predictions [{joy,neutral},{joy,neutral}]
    assert values["micro_f1"] == pytest.approx(4 / 7)
    assert values["macro_f1"] == pytest.approx(1 / 21)
    assert values["exact_match"] == 0.5
    assert values["failure_rate"] == 0
    before = store.rows("SELECT * FROM evaluation_items WHERE evaluation_id=?", (eid,))
    reparse(store, run, "labels-v2")
    assert (
        store.rows("SELECT * FROM evaluation_items WHERE evaluation_id=?", (eid,))
        == before
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute("DELETE FROM evaluation_items WHERE evaluation_id=?", (eid,))
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute(
            "UPDATE metric_values SET value=1 WHERE evaluation_id=?", (eid,)
        )
    exported = export_evaluation(store, eid, tmp_path / "export")
    assert (exported / "predictions.jsonl").read_text().count("\n") == 2
    assert (exported / "manifest.json").exists()


def test_failures_keep_denominator_and_partial_never_official(configured):
    store, config = configured
    config["max_attempts"] = 1
    run = prepare_run(store, config, counter=Counter())
    replies = iter([response(), None])

    def send(request):
        r = next(replies)
        if r is None:
            raise TimeoutError("offline fixture")
        return r

    execute_run(store, run, transport=send, counter=Counter())
    eid = evaluate(store, run, {})
    assert store.one(
        "SELECT value FROM metric_values WHERE evaluation_id=? AND metric_name='micro_f1' AND scope_key='overall'",
        (eid,),
    )["value"] == pytest.approx(0.8)
    assert (
        store.one("SELECT failed_items FROM evaluations WHERE evaluation_id=?", (eid,))[
            "failed_items"
        ]
        == 1
    )
    fresh = prepare_run(store, config, counter=Counter())
    partial = evaluate(store, fresh, {})
    assert (
        store.one("SELECT scope FROM evaluations WHERE evaluation_id=?", (partial,))[
            "scope"
        ]
        == "diagnostic_partial"
    )
    with pytest.raises(ValueError, match="official"):
        evaluate(store, fresh, {"scope": "official_full"})


def test_paired_bootstrap_same_predictions_zero_and_reject_mismatched_sets(configured):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())
    execute_run(store, run, transport=lambda request: response(), counter=Counter())
    first = evaluate(store, run, {})
    second = evaluate(store, run, {})
    comp = compare(store, first, second, seed=7, repeats=20)
    metric = store.one("SELECT * FROM metric_values WHERE comparison_id=?", (comp,))
    assert metric["value"] == metric["ci_low"] == metric["ci_high"] == 0
    different = evaluate(store, run, {"zero_division": 1})
    with pytest.raises(ValueError, match="scoring|评分"):
        compare(store, first, different, seed=7, repeats=20)
