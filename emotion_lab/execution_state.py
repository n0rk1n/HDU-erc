"""Short transactions for claims and lease-guarded finalization."""

from datetime import datetime, timedelta, timezone
from .storage import uid, now


def claim(store, item_id, timeout):
    lease = uid()
    expires = (
        (datetime.now(timezone.utc) + timedelta(seconds=timeout + 120))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    with store.transaction():
        item = store.one("SELECT * FROM run_items WHERE run_item_id=?", (item_id,))
        if item["status"] != "pending":
            return None
        store.transition(
            "run_items",
            "run_item_id",
            item_id,
            "running",
            worker_id="local-scheduler",
            lease_token=lease,
            lease_expires_at=expires,
            started_at=item["started_at"] or now(),
            completed_at=None,
        )
    return lease


def finalize(store, item_id, lease, prediction_id=None, error=None):
    with store.transaction():
        item = store.one("SELECT * FROM run_items WHERE run_item_id=?", (item_id,))
        if (
            item["lease_token"] != lease
            or item["status"] != "running"
            or item["lease_expires_at"] < now()
        ):
            return False
        if prediction_id:
            store.transition(
                "run_items",
                "run_item_id",
                item_id,
                "succeeded",
                final_prediction_id=prediction_id,
                completed_at=now(),
                failure_stage=None,
                failure_code=None,
                error_artifact_id=None,
                lease_token=None,
                lease_expires_at=None,
            )
        else:
            aid = store.put(error or {"code": "attempts_exhausted"}, "item_failure")
            store.transition(
                "run_items",
                "run_item_id",
                item_id,
                "failed",
                completed_at=now(),
                failure_stage="inference",
                failure_code=(error or {}).get("code", "attempts_exhausted"),
                error_artifact_id=aid,
                lease_token=None,
                lease_expires_at=None,
            )
            for step in store.rows(
                "SELECT step_id FROM execution_steps WHERE run_item_id=? AND status='running'",
                (item_id,),
            ):
                store.transition(
                    "execution_steps",
                    "step_id",
                    step["step_id"],
                    "failed",
                    completed_at=now(),
                    error_artifact_id=aid,
                    trace_complete=1,
                )
    return True


def recover(store, run_id):
    # Caller owns the process-level scheduler lock, so no old local worker can still commit.
    from .calls import save_response
    from .llm.transport import Response

    config = store.json(
        store.one(
            "SELECT method_config_artifact_id FROM experiment_runs WHERE run_id=?",
            (run_id,),
        )["method_config_artifact_id"]
    )
    captured = store.rows(
        "SELECT * FROM v_call_audit WHERE run_id=? AND response_artifact_id IS NOT NULL AND status!='response_received'",
        (run_id,),
    )
    for a in captured:
        save_response(
            store,
            a["call_attempt_id"],
            Response(
                a["http_status"],
                store.json(a["response_metadata_artifact_id"]),
                store.read(a["response_artifact_id"]),
            ),
            a["latency_ms"] or 0,
            config,
        )
    with store.transaction():
        attempts = store.rows(
            "SELECT a.* FROM call_attempts a JOIN execution_steps s USING(step_id) JOIN run_items i USING(run_item_id) WHERE i.run_id=? AND a.status IN ('prepared','dispatched')",
            (run_id,),
        )
        for a in attempts:
            status = (
                "cancelled_before_dispatch"
                if a["status"] == "prepared"
                else "outcome_unknown"
            )
            store.transition(
                "call_attempts",
                "call_attempt_id",
                a["call_attempt_id"],
                status,
                completed_at=now(),
                billing_state="not_dispatched"
                if status == "cancelled_before_dispatch"
                else "unknown",
            )
        for item in store.rows(
            "SELECT * FROM run_items WHERE run_id=? AND status IN ('running','failed','cancelled')",
            (run_id,),
        ):
            store.transition(
                "run_items",
                "run_item_id",
                item["run_item_id"],
                "pending",
                lease_token=None,
                lease_expires_at=None,
            )


def cancel_run(store, run_id):
    # CLI obtains the same scheduling lock before calling this operation.
    with store.transaction():
        for item in store.rows(
            "SELECT run_item_id FROM run_items WHERE run_id=? AND status IN ('pending','running')",
            (run_id,),
        ):
            store.transition(
                "run_items",
                "run_item_id",
                item["run_item_id"],
                "cancelled",
                completed_at=now(),
                lease_token=None,
                lease_expires_at=None,
            )
        store.transition(
            "experiment_runs",
            "run_id",
            run_id,
            "cancelled",
            completed_at=now(),
            stop_reason="user_cancelled",
        )
