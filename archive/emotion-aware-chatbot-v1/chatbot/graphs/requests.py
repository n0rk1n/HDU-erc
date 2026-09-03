"""Canonical durable request bindings for idempotent graph operations."""

from __future__ import annotations

import hashlib
import json


def request_fingerprint(operation: str, **fields: str) -> str:
    payload = {"operation": operation, **fields}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def turn_fingerprint(content: str) -> str:
    return request_fingerprint("turn", content=content.strip())


def regeneration_fingerprint(target_message_id: str, reason: str) -> str:
    return request_fingerprint(
        "regenerate", target_message_id=target_message_id, reason=reason
    )


def request_binding_matches(result: dict, operation: str, fingerprint: str) -> bool:
    return (
        result.get("operation") == operation
        and result.get("input_fingerprint") == fingerprint
    )
