"""Persist raw provider responses before parsing; retain every attempt."""

import json
from decimal import Decimal, ROUND_HALF_UP

from .storage import digest, now
from .taxonomy import parse_content
from .llm.transport import safe_facts


def token_value(value):
    return value if type(value) is int and value >= 0 else None


def response_fields(body):
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return {}, None, {}, None, None
    except (ValueError, UnicodeError):
        return {}, None, {}, None, None
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    choices = data.get("choices")
    choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    reasoning = message.get("reasoning_content")
    return (
        data,
        message.get("content"),
        usage,
        reasoning if isinstance(reasoning, str) else None,
        choice.get("finish_reason"),
    )


def estimate_cost(usage, pricing):
    if not pricing:
        return None, None, None
    inp = token_value(usage.get("prompt_tokens"))
    out = token_value(usage.get("completion_tokens"))
    if inp is None or out is None:
        return None, None, None
    cached = token_value(usage.get("prompt_cache_hit_tokens"))
    details = usage.get("prompt_tokens_details")
    if cached is None and isinstance(details, dict):
        cached = token_value(details.get("cached_tokens"))
    if "cached_input_per_million" in pricing and cached is None:
        return None, None, None
    cached = cached or 0
    if cached > inp:
        return None, None, None
    # rates are currency / 1M tokens, so tokens * rate already yields microcurrency.
    cost = (
        Decimal(inp - cached) * Decimal(str(pricing["input_per_million"]))
        + Decimal(out) * Decimal(str(pricing["output_per_million"]))
        + Decimal(cached)
        * Decimal(
            str(pricing.get("cached_input_per_million", pricing["input_per_million"]))
        )
    )
    return (
        int(cost.quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
        pricing["currency"],
        {
            "input": inp,
            "cached_input": cached,
            "output": out,
            "rates": pricing,
            "rounding": "half-up-microcurrency",
        },
    )


def save_response(store, attempt_id, response, elapsed, config):
    attempt = store.one(
        "SELECT * FROM call_attempts WHERE call_attempt_id=?", (attempt_id,)
    )
    if attempt["response_artifact_id"] is not None:
        if (
            attempt["status"] == "response_received"
            or store.read(attempt["response_artifact_id"]) != response.body
        ):
            raise ValueError("response already captured")
        response_id = attempt["response_artifact_id"]
        metadata_id = attempt["response_metadata_artifact_id"]
    else:
        # Raw evidence commits before any parsing or cost calculation can fail.
        with store.transaction():
            response_id = store.put(
                response.body,
                "model_response",
                media_type=response.headers.get("content-type", "application/json"),
                redaction="credentials-v1",
                capture="http-response-body",
            )
            metadata_id = store.put(
                response.headers, "response_metadata", redaction="credentials-v1"
            )
            store.db.execute(
                "UPDATE call_attempts SET response_artifact_id=?,response_metadata_artifact_id=?,http_status=?,first_response_at=? WHERE call_attempt_id=?",
                (response_id, metadata_id, response.status, now(), attempt_id),
            )
            store.event("call_attempt_id", attempt_id, "raw_response_saved")
    data, _, usage, reasoning, finish = response_fields(response.body)
    details = usage.get("prompt_tokens_details") or {}
    output_details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    if not isinstance(output_details, dict):
        output_details = {}
    cost, currency, calculation = estimate_cost(usage, config.get("pricing"))
    with store.transaction():
        http_error = (
            store.put(
                {"http_status": response.status, "response_artifact_id": response_id},
                "http_error",
            )
            if not 200 <= response.status < 300
            else None
        )
        error = attempt["error_artifact_id"] or http_error
        if http_error:
            store.event(
                "call_attempt_id", attempt_id, "http_error_received", payload=http_error
            )
        store.transition(
            "call_attempts",
            "call_attempt_id",
            attempt_id,
            "response_received",
            response_artifact_id=response_id,
            reasoning_artifact_id=store.put(reasoning, "provider_reasoning")
            if reasoning is not None
            else None,
            error_artifact_id=error,
            response_metadata_artifact_id=metadata_id,
            usage_artifact_id=store.put(usage, "provider_usage") if usage else None,
            http_status=response.status,
            provider_request_id=str(
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
                or data.get("id")
            )
            if (
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
                or isinstance(data.get("id"), str)
            )
            else None,
            resolved_model=data.get("model")
            if isinstance(data.get("model"), str)
            else None,
            finish_reason=finish if isinstance(finish, str) else None,
            first_response_at=now(),
            completed_at=now(),
            latency_ms=elapsed,
            input_tokens=token_value(usage.get("prompt_tokens")),
            output_tokens=token_value(usage.get("completion_tokens")),
            total_tokens=token_value(usage.get("total_tokens")),
            cached_input_tokens=token_value(
                usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
            ),
            reasoning_tokens=token_value(output_details.get("reasoning_tokens")),
            usage_source="provider" if usage else "unavailable",
            billing_state="estimated" if cost is not None else "unknown",
            estimated_cost_micros=cost,
            currency=currency,
            cost_calculation_artifact_id=store.put(calculation, "cost_calculation")
            if calculation
            else None,
        )


def save_transport_error(store, attempt_id, exc, elapsed, secret=""):
    with store.transaction():
        error = store.put(
            safe_facts({"type": type(exc).__name__, "message": str(exc)}, secret),
            "transport_error",
            redaction="credentials-v1",
        )
        store.transition(
            "call_attempts",
            "call_attempt_id",
            attempt_id,
            "outcome_unknown",
            error_artifact_id=error,
            completed_at=now(),
            latency_ms=elapsed,
            billing_state="unknown",
        )
        store.event(
            "call_attempt_id", attempt_id, "transport_error_details", payload=error
        )


def parse_attempt(store, attempt_id, parser_version="labels-v1"):
    old = store.rows(
        "SELECT prediction_id FROM predictions WHERE call_attempt_id=? AND parser_version=? AND output_schema_version=?",
        (attempt_id, parser_version, "native-labels-v1"),
    )
    if old:
        return old[0]["prediction_id"]
    a = store.one(
        "SELECT a.*,s.run_item_id,i.sample_id,r.taxonomy_config_artifact_id,r.method_config_artifact_id FROM call_attempts a JOIN execution_steps s USING(step_id) JOIN run_items i USING(run_item_id) JOIN experiment_runs r USING(run_id) WHERE call_attempt_id=?",
        (attempt_id,),
    )
    body = store.read(a["response_artifact_id"])
    _, content, _, _, _ = response_fields(body)
    taxonomy = store.json(a["taxonomy_config_artifact_id"])
    config = store.json(a["method_config_artifact_id"])
    parsed, errors = parse_content(
        content, taxonomy["labels"], config["require_evidence"]
    )
    if not a["http_status"] or not 200 <= a["http_status"] < 300:
        errors.append({"code": "http_error", "status": a["http_status"]})
    ds = store.one(
        "SELECT dataset_version_id,raw_text FROM samples WHERE sample_id=?",
        (a["sample_id"],),
    )
    if (
        config["require_evidence"]
        and isinstance(parsed, dict)
        and isinstance(parsed.get("evidence"), dict)
    ):
        if any(
            not isinstance(v, str) or v not in ds["raw_text"]
            for v in parsed["evidence"].values()
        ):
            errors.append({"code": "evidence_not_in_input"})
    labels = store.rows(
        "SELECT label_id,label_name FROM label_definitions WHERE dataset_version_id=?",
        (ds["dataset_version_id"],),
    )
    ids = {r["label_name"]: r["label_id"] for r in labels}
    with store.transaction():
        pred = store.insert(
            "predictions",
            run_item_id=a["run_item_id"],
            call_attempt_id=attempt_id,
            purpose="final",
            parser_version=parser_version,
            output_schema_version="native-labels-v1",
            source_response_sha256=digest(body),
            parsed_artifact_id=store.put(parsed, "parsed_output")
            if parsed is not None
            else None,
            evidence_artifact_id=store.put(parsed["evidence"], "prediction_evidence")
            if isinstance(parsed, dict) and parsed.get("evidence") is not None
            else None,
            validation_artifact_id=store.put(
                {
                    "errors": errors,
                    "parser_version": parser_version,
                    "implementation": "strict-native-labels-v1",
                },
                "validation",
            ),
            status="invalid" if errors else "valid",
        )
        if not errors:
            for position, label in enumerate(parsed["labels"]):
                score = parsed.get("scores", {}).get(label)
                store.insert(
                    "prediction_labels",
                    prediction_id=pred,
                    dataset_version_id=ds["dataset_version_id"],
                    label_id=ids[label],
                    output_position=position,
                    reported_score=score,
                    score_kind="model_reported" if score is not None else None,
                    evidence_json={"quote": parsed["evidence"][label]}
                    if isinstance(parsed.get("evidence"), dict)
                    and label in parsed["evidence"]
                    else None,
                )
        store.event(
            "call_attempt_id",
            attempt_id,
            "parsed",
            payload=store.put(
                {"prediction_id": pred, "valid": not errors}, "parse_result"
            ),
        )
    return pred


def reparse(store, run_id, parser_version):
    attempts = store.rows(
        "SELECT a.call_attempt_id FROM call_attempts a JOIN execution_steps s USING(step_id) JOIN run_items i USING(run_item_id) WHERE i.run_id=? AND a.response_artifact_id IS NOT NULL ORDER BY i.ordinal,a.attempt_no",
        (run_id,),
    )
    return [
        parse_attempt(store, a["call_attempt_id"], parser_version) for a in attempts
    ]
