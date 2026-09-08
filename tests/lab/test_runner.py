import json
import sqlite3

import pytest

from emotion_lab.datasets import make_set
from emotion_lab.runner import prepare_run, execute_run, reparse
from emotion_lab.llm.transport import Response


class Counter:
    identity = "offline-test-counter"
    version = "fixture-v1"

    def count(self, messages):
        return len(json.dumps(messages))


@pytest.fixture
def configured(lab):
    store, ds = lab
    corpus = make_set(
        store,
        {
            "dataset_version_id": ds,
            "name": "train",
            "purpose": "corpus",
            "split": "train",
        },
    )
    dev = make_set(
        store,
        {
            "dataset_version_id": ds,
            "name": "dev",
            "purpose": "dev_pilot",
            "split": "dev",
        },
    )
    config = {
        "name": "offline test",
        "corpus_set_id": corpus,
        "evaluation_set_id": dev,
        "method": "zero-shot",
        "seed": 42,
        "k": 0,
        "max_attempts": 2,
        "concurrency": 1,
        "model": {
            "name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "temperature": 0,
            "timeout_seconds": 10,
            "max_tokens": 100,
            "context_tokens": 10000,
            "tokenizer_model": "deepseek-v4-flash",
            "thinking": "disabled",
        },
    }
    return store, config


def response(labels=("joy", "neutral"), **kwargs):
    data = {
        "id": "request-1",
        "model": "resolved-fixture",
        "choices": [
            {
                "message": {"content": json.dumps({"labels": labels})},
                "finish_reason": "stop",
            }
        ],
        **kwargs,
    }
    return Response(200, {"x-request-id": "rid-1"}, json.dumps(data).encode())


def test_dry_run_is_read_only_and_has_no_predictions(configured):
    store, config = configured
    before = store.db.total_changes
    preview = prepare_run(store, config, dry_run=True, counter=Counter())
    assert preview["sample_count"] == 2
    assert store.db.total_changes == before
    assert not store.rows("SELECT * FROM call_attempts")


def test_multilabel_response_raw_usage_and_reparse(configured):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())
    execute_run(store, run, transport=lambda request: response(), counter=Counter())
    assert (
        store.one("SELECT status FROM experiment_runs WHERE run_id=?", (run,))["status"]
        == "completed"
    )
    attempts = store.rows("SELECT * FROM call_attempts")
    assert len(attempts) == 2
    assert attempts[0]["total_tokens"] is None
    assert attempts[0]["estimated_cost_micros"] is None
    assert (
        json.loads(store.read(attempts[0]["request_artifact_id"]))["messages"][-1][
            "role"
        ]
        == "user"
    )
    assert (
        json.loads(store.read(attempts[0]["response_artifact_id"]))["model"]
        == "resolved-fixture"
    )
    assert store.one("SELECT COUNT(*) n FROM prediction_labels")["n"] == 4
    reparse(store, run, "labels-v2")
    assert store.one("SELECT COUNT(*) n FROM predictions")["n"] == 4
    assert store.one("SELECT COUNT(*) n FROM call_attempts")["n"] == 2
    with pytest.raises(sqlite3.IntegrityError):
        store.db.execute("UPDATE call_attempts SET resolved_model='overwritten'")


def test_invalid_output_retry_preserves_both_attempts(configured):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())
    replies = iter([response(("unknown",)), response(), response(), response()])
    execute_run(store, run, transport=lambda request: next(replies), counter=Counter())
    assert len(store.rows("SELECT * FROM call_attempts")) == 3
    assert len(store.rows("SELECT * FROM predictions WHERE status='invalid'")) == 1
    assert (
        store.one('SELECT COUNT(*) n FROM run_items WHERE status="succeeded"')["n"] == 2
    )


@pytest.mark.parametrize(
    "point", ["prepared", "dispatched", "response_saved", "parsed", "before_finalize"]
)
def test_crash_resume_preserves_evidence_and_successes(configured, point):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())

    class Crash(BaseException):
        pass

    def fail(stage, attempt):
        if stage == point:
            raise Crash()

    with pytest.raises(Crash):
        execute_run(
            store,
            run,
            transport=lambda request: response(),
            counter=Counter(),
            checkpoint=fail,
        )
    execute_run(
        store, run, transport=lambda request: response(), counter=Counter(), resume=True
    )
    assert (
        store.one("SELECT status FROM experiment_runs WHERE run_id=?", (run,))["status"]
        == "completed"
    )
    count = len(store.rows("SELECT * FROM call_attempts"))
    execute_run(
        store,
        run,
        transport=lambda request: pytest.fail("must skip successes"),
        counter=Counter(),
        resume=True,
    )
    assert len(store.rows("SELECT * FROM call_attempts")) == count
    if point == "dispatched":
        assert store.rows("SELECT * FROM call_attempts WHERE status='outcome_unknown'")


def test_http_error_and_timeout_are_not_zero_cost(configured):
    store, config = configured
    run = prepare_run(store, config, counter=Counter())

    def timeout(request):
        raise TimeoutError("provider timed out")

    execute_run(store, run, transport=timeout, counter=Counter())
    assert store.one('SELECT COUNT(*) n FROM run_items WHERE status="failed"')["n"] == 2
    assert all(
        a["estimated_cost_micros"] is None
        for a in store.rows("SELECT * FROM call_attempts")
    )
    assert all(
        a["status"] == "outcome_unknown"
        for a in store.rows("SELECT * FROM call_attempts")
    )


def test_unsupported_method_never_silently_falls_back(configured):
    store, config = configured
    config["method"] = "contrastive"
    with pytest.raises(ValueError, match="not implemented|未实现"):
        prepare_run(store, config, counter=Counter())


def test_lexical_candidates_are_train_only_and_audited(configured):
    store, config = configured
    config.update(method="lexical", k=1, candidate_count=2)
    run = prepare_run(store, config, counter=Counter())
    execute_run(store, run, transport=lambda request: response(), counter=Counter())
    candidates = store.rows(
        "SELECT r.*,s.split FROM retrieval_items r JOIN samples s USING(sample_id)"
    )
    assert len(candidates) == 4
    assert all(r["split"] == "train" for r in candidates)
    assert sum(r["decision"] == "selected" for r in candidates) == 2
    assert (
        len(
            store.rows(
                "SELECT * FROM execution_steps WHERE kind='retrieve' AND output_artifact_id IS NOT NULL"
            )
        )
        == 2
    )
