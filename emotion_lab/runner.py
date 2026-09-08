"""One SQLite writer, bounded concurrent HTTP calls, explicit recovery boundaries."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import time

from .calls import parse_attempt, reparse, save_response, save_transport_error
from .config import credentials
from .execution_state import claim, finalize, recover, cancel_run
from .llm.counter import TokenCounter
from .llm.parameters import thinking_extra_body
from .llm.transport import HTTPTransport, safe_facts
from .preparation import prepare_run, application_sources
from .retrieval import Retriever
from .storage import digest, json_bytes, now


__all__ = ["prepare_run", "execute_run", "reparse", "cancel_run", "scheduler_lock"]


@contextmanager
def scheduler_lock(store):
    with (store.root / "scheduler.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another scheduler is active") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _step(store, item, kind, config_artifact, input_artifact):
    existing = store.rows(
        "SELECT * FROM execution_steps WHERE run_item_id=? AND step_key=?", (item, kind)
    )
    if existing:
        return existing[0]["step_id"]
    with store.transaction():
        step = store.insert(
            "execution_steps",
            run_item_id=item,
            step_key=kind,
            step_no=1 if kind == "retrieve" else 2,
            kind=kind,
            implementation_version="emotion-lab-v1",
            config_artifact_id=config_artifact,
            input_artifact_id=input_artifact,
            status="running",
            started_at=now(),
            trace_complete=0,
        )
        store.event("step_id", step, "started", None, "running")
    return step


def _request_for(store, item, run, config, taxonomy, retriever, counter):
    existing = store.rows(
        "SELECT * FROM execution_steps WHERE run_item_id=? AND kind='classify'",
        (item["run_item_id"],),
    )
    if existing:
        return existing[0]["step_id"], store.read(existing[0]["input_artifact_id"])
    query = store.one(
        "SELECT sample_id,raw_text,text_sha256,normalized_text_sha256 FROM samples WHERE sample_id=?",
        (item["sample_id"],),
    )
    started = time.monotonic()
    _, candidates, trace, messages = retriever.select(query, config, taxonomy, counter)
    elapsed = int((time.monotonic() - started) * 1000)
    model = config["model"]
    body = {
        "model": model["name"],
        "messages": messages,
        "temperature": model["temperature"],
        "max_tokens": model["max_tokens"],
        "stream": False,
    }
    body.update(
        thinking_extra_body(
            model["name"],
            model["base_url"],
            model["thinking"],
            model.get("reasoning_effort"),
        )
        or {}
    )
    request = json_bytes(body)
    with store.transaction():
        query_id = store.put(
            {"sample_id": query["sample_id"], "text": query["raw_text"]}, "query_input"
        )
        request_id = store.put(
            request,
            "model_request",
            media_type="application/json",
            capture="http-request-body",
            redaction="credentials-v1",
        )
    if config["method"] != "zero-shot":
        step = _step(
            store,
            item["run_item_id"],
            "retrieve",
            run["method_config_artifact_id"],
            query_id,
        )
        if (
            store.one("SELECT status FROM execution_steps WHERE step_id=?", (step,))[
                "status"
            ]
            != "completed"
        ):
            with store.transaction():
                for candidate in candidates:
                    store.insert("retrieval_items", retrieval_step_id=step, **candidate)
                store.transition(
                    "execution_steps",
                    "step_id",
                    step,
                    "completed",
                    output_artifact_id=store.put(
                        trace, "retrieval_trace", inline=False
                    ),
                    completed_at=now(),
                    latency_ms=elapsed,
                    trace_complete=1,
                )
    step = _step(
        store,
        item["run_item_id"],
        "classify",
        run["method_config_artifact_id"],
        request_id,
    )
    return step, request


def _valid_existing(store, step, parser):
    for a in store.rows(
        "SELECT call_attempt_id FROM call_attempts WHERE step_id=? AND response_artifact_id IS NOT NULL ORDER BY attempt_no",
        (step,),
    ):
        pred = parse_attempt(store, a["call_attempt_id"], parser)
        if (
            store.one("SELECT status FROM predictions WHERE prediction_id=?", (pred,))[
                "status"
            ]
            == "valid"
        ):
            return pred
    return None


def execute_run(
    store, run_id, transport=None, counter=None, resume=False, checkpoint=None
):
    checkpoint = checkpoint or (lambda stage, attempt: None)
    with scheduler_lock(store):
        run = store.one("SELECT * FROM experiment_runs WHERE run_id=?", (run_id,))
        if run["status"] == "completed":
            return run_id
        if not resume and run["status"] != "ready":
            raise ValueError("run already started; use resume")
        config = store.json(run["method_config_artifact_id"])
        if digest(config) != run["config_sha256"]:
            raise ValueError("configuration fingerprint mismatch")
        expected_sources = store.json(run["environment_artifact_id"])["source_hashes"]
        if expected_sources != {
            k: digest(v.encode()) for k, v in application_sources().items()
        }:
            raise ValueError(
                "application code changed; create a new run or restore the recorded code"
            )
        taxonomy = store.json(run["taxonomy_config_artifact_id"])
        counter = counter or TokenCounter(config["model"]["tokenizer_model"])
        if config["counter"] != {
            "identity": counter.identity,
            "version": counter.version,
        }:
            raise ValueError("token counter version changed")
        own_transport = transport is None
        if own_transport:
            transport = HTTPTransport(config["model"], credentials())
        retriever = Retriever(store, config["corpus_set_id"])
        if resume:
            recover(store, run_id)
        with store.transaction():
            store.transition(
                "experiment_runs",
                "run_id",
                run_id,
                "running",
                started_at=run["started_at"] or now(),
                completed_at=None,
                stop_reason=None,
            )
        pending = deque(
            store.rows(
                "SELECT * FROM run_items WHERE run_id=? AND status='pending' ORDER BY ordinal",
                (run_id,),
            )
        )
        futures = {}
        used = store.one(
            "SELECT COUNT(*) n FROM v_call_audit WHERE run_id=? AND status!='cancelled_before_dispatch'",
            (run_id,),
        )["n"]
        stopped = False

        def submit(pool, item, lease=None):
            nonlocal used, stopped
            item_id = item["run_item_id"]
            lease = lease or claim(store, item_id, config["model"]["timeout_seconds"])
            if lease is None:
                return
            try:
                step, request = _request_for(
                    store, item, run, config, taxonomy, retriever, counter
                )
                prediction = _valid_existing(store, step, config["parser_version"])
                if prediction:
                    if finalize(store, item_id, lease, prediction):
                        with store.transaction():
                            store.transition(
                                "execution_steps",
                                "step_id",
                                step,
                                "completed",
                                completed_at=now(),
                                trace_complete=1,
                                output_artifact_id=store.put(
                                    {"prediction_id": prediction},
                                    "classification_result",
                                ),
                            )
                    return
                attempts = store.rows(
                    "SELECT * FROM call_attempts WHERE step_id=? ORDER BY attempt_no",
                    (step,),
                )
                count = sum(
                    a["status"] != "cancelled_before_dispatch" for a in attempts
                )
                if count >= config["max_attempts"]:
                    finalize(
                        store, item_id, lease, error={"code": "attempts_exhausted"}
                    )
                    return
                if used >= config["max_total_attempts"]:
                    stopped = True
                    with store.transaction():
                        store.transition(
                            "run_items",
                            "run_item_id",
                            item_id,
                            "pending",
                            lease_token=None,
                            lease_expires_at=None,
                        )
                    return
                with store.transaction():
                    request_id = store.one(
                        "SELECT input_artifact_id FROM execution_steps WHERE step_id=?",
                        (step,),
                    )["input_artifact_id"]
                    aid = store.insert(
                        "call_attempts",
                        step_id=step,
                        attempt_no=max((a["attempt_no"] for a in attempts), default=0)
                        + 1,
                        retry_of_attempt_id=attempts[-1]["call_attempt_id"]
                        if attempts
                        else None,
                        transport_kind="remote",
                        provider="openai-compatible",
                        requested_model=config["model"]["name"],
                        endpoint=config["model"]["base_url"].rstrip("/")
                        + "/chat/completions",
                        request_artifact_id=request_id,
                        request_sha256=digest(request),
                        status="prepared",
                        prepared_at=now(),
                        usage_source="unavailable",
                        billing_state="not_dispatched",
                        price_snapshot_artifact_id=run["price_snapshot_artifact_id"],
                    )
                    store.event("call_attempt_id", aid, "prepared", None, "prepared")
                checkpoint("prepared", aid)
                with store.transaction():
                    from datetime import datetime, timedelta, timezone

                    expires = (
                        (
                            datetime.now(timezone.utc)
                            + timedelta(
                                seconds=config["model"]["timeout_seconds"] + 120
                            )
                        )
                        .isoformat(timespec="microseconds")
                        .replace("+00:00", "Z")
                    )
                    changed = store.db.execute(
                        "UPDATE run_items SET lease_expires_at=? WHERE run_item_id=? AND lease_token=? AND status='running'",
                        (expires, item_id, lease),
                    ).rowcount
                    if changed != 1:
                        raise ValueError("lease lost before dispatch")
                    store.transition(
                        "call_attempts",
                        "call_attempt_id",
                        aid,
                        "dispatched",
                        dispatch_started_at=now(),
                        billing_state="unknown",
                    )
                used += 1
                checkpoint("dispatched", aid)
                started = time.monotonic()
                futures[pool.submit(transport, request)] = (
                    item,
                    lease,
                    step,
                    aid,
                    started,
                )
            except Exception as exc:
                finalize(
                    store,
                    item_id,
                    lease,
                    error={
                        "code": "preflight_error",
                        "message": safe_facts(str(exc), getattr(transport, "key", "")),
                    },
                )

        try:
            with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                while pending or futures:
                    while (
                        pending and len(futures) < config["concurrency"] and not stopped
                    ):
                        submit(pool, pending.popleft())
                    if not futures:
                        break
                    done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        item, lease, step, aid, started = futures.pop(future)
                        elapsed = int((time.monotonic() - started) * 1000)
                        try:
                            response = future.result()
                        except Exception as exc:
                            save_transport_error(
                                store, aid, exc, elapsed, getattr(transport, "key", "")
                            )
                            submit(pool, item, lease)
                            continue
                        save_response(store, aid, response, elapsed, config)
                        checkpoint("response_saved", aid)
                        pred = parse_attempt(store, aid, config["parser_version"])
                        checkpoint("parsed", aid)
                        if (
                            store.one(
                                "SELECT status FROM predictions WHERE prediction_id=?",
                                (pred,),
                            )["status"]
                            == "valid"
                        ):
                            checkpoint("before_finalize", aid)
                            if finalize(store, item["run_item_id"], lease, pred):
                                with store.transaction():
                                    store.transition(
                                        "execution_steps",
                                        "step_id",
                                        step,
                                        "completed",
                                        completed_at=now(),
                                        trace_complete=1,
                                        output_artifact_id=store.put(
                                            {"prediction_id": pred},
                                            "classification_result",
                                        ),
                                    )
                        else:
                            submit(pool, item, lease)
            states = store.rows(
                "SELECT status,COUNT(*) n FROM run_items WHERE run_id=? GROUP BY status",
                (run_id,),
            )
            counts = {s["status"]: s["n"] for s in states}
            status = (
                "paused"
                if counts.get("pending") or counts.get("running")
                else ("completed_with_errors" if counts.get("failed") else "completed")
            )
            with store.transaction():
                store.transition(
                    "experiment_runs",
                    "run_id",
                    run_id,
                    status,
                    completed_at=None if status == "paused" else now(),
                    stop_reason="attempt_budget" if stopped else None,
                )
        except BaseException:
            with store.transaction():
                store.transition(
                    "experiment_runs",
                    "run_id",
                    run_id,
                    "paused",
                    stop_reason="interrupted",
                )
            raise
        finally:
            if own_transport:
                transport.close()
    return run_id
