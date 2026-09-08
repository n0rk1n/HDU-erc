import sqlite3
import pytest

from emotion_lab.runner import prepare_run, execute_run
from emotion_lab.execution_state import claim, finalize
from emotion_lab.calls import save_response, estimate_cost
from emotion_lab.llm.transport import Response
from test_runner import Counter, response


def test_cross_item_final_and_late_label_insertion_are_rejected(configured):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())
    execute_run(store, run, transport=lambda request: response(), counter=Counter())
    items = store.rows("SELECT * FROM run_items ORDER BY ordinal")
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute(
            "UPDATE run_items SET final_prediction_id=? WHERE run_item_id=?",
            (items[1]["final_prediction_id"], items[0]["run_item_id"]),
        )
    ds = store.one("SELECT dataset_version_id FROM samples LIMIT 1")[
        "dataset_version_id"
    ]
    with pytest.raises(sqlite3.IntegrityError):
        store.insert(
            "prediction_labels",
            prediction_id=items[0]["final_prediction_id"],
            dataset_version_id=ds,
            label_id=0,
            output_position=2,
        )


def test_stale_worker_cannot_finish_new_lease(configured):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())
    item = store.one(
        "SELECT run_item_id FROM run_items WHERE run_id=? ORDER BY ordinal LIMIT 1",
        (run,),
    )["run_item_id"]
    old = claim(store, item, 10)
    store.db.execute(
        "UPDATE run_items SET lease_token='new-worker' WHERE run_item_id=?", (item,)
    )
    assert not finalize(store, item, old, error={"code": "old-worker"})
    assert (
        store.one("SELECT status FROM run_items WHERE run_item_id=?", (item,))["status"]
        == "running"
    )


def test_malformed_usage_preserves_raw_response(configured):
    store, config = configured
    config["pricing"] = {
        "currency": "USD",
        "input_per_million": 1,
        "output_per_million": 2,
        "cached_input_per_million": 0.5,
    }
    run = prepare_run(store, config, counter=Counter())
    reply = response(
        usage={
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "prompt_tokens_details": ["malformed"],
        }
    )
    execute_run(store, run, transport=lambda request: reply, counter=Counter())
    assert (
        store.one("SELECT status FROM experiment_runs WHERE run_id=?", (run,))["status"]
        == "completed"
    )
    attempts = store.rows("SELECT * FROM call_attempts")
    assert len(attempts) == 2
    assert all(store.read(a["response_artifact_id"]) == reply.body for a in attempts)
    assert all(a["estimated_cost_micros"] is None for a in attempts)


def test_cost_does_not_double_count_reasoning_or_cache():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_cache_hit_tokens": 60,
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    cost, currency, _ = estimate_cost(
        usage,
        {
            "currency": "USD",
            "input_per_million": "1",
            "cached_input_per_million": ".5",
            "output_per_million": "2",
        },
    )
    assert cost == 110
    assert currency == "USD"


def test_verify_detects_changed_frozen_set_fingerprint(configured):
    store, config = configured
    # Simulate external corruption after removing the corresponding protection trigger.
    store.db.execute("DROP TRIGGER frozen_sample_sets_UPDATE")
    store.db.execute("UPDATE sample_sets SET members_sha256='incorrect'")
    assert not store.verify()["ok"]


def test_http_500_records_body_separately_from_invalid_prediction(configured):
    store, config = configured
    config["max_attempts"] = 1
    run = prepare_run(store, config, counter=Counter())
    reply = Response(
        500,
        {"x-request-id": "error-request"},
        b'{"error":{"message":"offline server error"}}',
    )
    execute_run(store, run, transport=lambda request: reply, counter=Counter())
    attempts = store.rows("SELECT * FROM call_attempts")
    assert all(
        a["status"] == "response_received" and a["http_status"] == 500 for a in attempts
    )
    assert all(store.read(a["response_artifact_id"]) == reply.body for a in attempts)
    assert (
        store.one('SELECT COUNT(*) n FROM predictions WHERE status="invalid"')["n"] == 2
    )


def test_retrieval_is_reproducible_after_fresh_database_import(configured, tmp_path):
    from emotion_lab.storage import Store
    from emotion_lab.datasets import import_goemotions, make_set
    from emotion_lab.retrieval import Retriever
    from emotion_lab.taxonomy import load_taxonomy
    from emotion_lab.config import validate_config
    from test_storage import source_fixture

    store, config = configured
    config.update(method="random", k=1, candidate_count=2)
    config = validate_config(config)
    other = Store(tmp_path / "other", initialize=True)
    try:
        ds = import_goemotions(other, source_fixture(tmp_path / "source2"))
        corpus = make_set(
            other,
            {
                "name": "train",
                "purpose": "corpus",
                "split": "train",
                "dataset_version_id": ds,
            },
        )

        def selected(s, cid):
            query = s.one("SELECT * FROM samples WHERE source_id='d1'")
            retriever = Retriever(s, cid)
            chosen, *_ = retriever.select(query, config, load_taxonomy(), Counter())
            return [r["source_id"] for r in chosen]

        # Several seeds exercise the ordering contract independently of random internal UUIDs.
        for seed in range(10):
            config["seed"] = seed
            assert selected(store, config["corpus_set_id"]) == selected(other, corpus)
    finally:
        other.close()


def test_late_response_preserves_timeout_error_and_frozen_evaluation(configured):
    from emotion_lab.evaluation import evaluate

    store, config = configured
    config["max_attempts"] = 1
    run = prepare_run(store, config, counter=Counter())

    def timeout(request):
        raise TimeoutError("lost response")

    execute_run(store, run, transport=timeout, counter=Counter())
    eid = evaluate(store, run, {})
    before = store.rows("SELECT * FROM evaluation_items WHERE evaluation_id=?", (eid,))
    a = store.one("SELECT * FROM call_attempts ORDER BY created_at LIMIT 1")
    error_id = a["error_artifact_id"]
    save_response(
        store, a["call_attempt_id"], Response(500, {}, b'{"error":"late"}'), 1, config
    )
    after = store.one(
        "SELECT * FROM call_attempts WHERE call_attempt_id=?", (a["call_attempt_id"],)
    )
    assert after["error_artifact_id"] == error_id
    assert (
        store.rows("SELECT * FROM evaluation_items WHERE evaluation_id=?", (eid,))
        == before
    )


def test_retry_recovery_finishes_step_and_trace(configured):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())

    class Crash(BaseException):
        pass

    def interrupt(stage, attempt):
        if stage == "parsed":
            raise Crash()

    with pytest.raises(Crash):
        execute_run(
            store,
            run,
            transport=lambda r: response(),
            counter=Counter(),
            checkpoint=interrupt,
        )
    execute_run(
        store, run, transport=lambda r: response(), counter=Counter(), resume=True
    )
    assert all(
        s["status"] == "completed" and s["trace_complete"] == 1
        for s in store.rows("SELECT * FROM execution_steps WHERE kind='classify'")
    )


def test_resume_rejects_changed_application_code(configured, monkeypatch):
    import emotion_lab.runner as runner

    store, config = configured
    run = prepare_run(store, config, counter=Counter())
    monkeypatch.setattr(
        runner,
        "application_sources",
        lambda: {"changed.py": "different implementation"},
        raising=False,
    )
    with pytest.raises(ValueError, match="code|代码"):
        execute_run(store, run, transport=lambda r: response(), counter=Counter())
