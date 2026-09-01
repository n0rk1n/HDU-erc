"""Emotion-correctness feedback persistence."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Coroutine
from uuid import uuid4

from langgraph.store.base import BaseStore

from chatbot.core.runtime_store import DEFAULT_RUNTIME_DB_PATH, RuntimeStore

RUNTIME_DB_PATH = DEFAULT_RUNTIME_DB_PATH
EMOTION_FEEDBACK_NAMESPACE = "emotion_feedback"
ALLOWED_EMOTION_FEEDBACK = {"accurate", "too_positive", "too_negative", "wrong_emotion"}
_FEEDBACK_LOCK = threading.RLock()


async def _load_emotion_feedback(
    store: BaseStore,
    client_id: str,
    thread_id: str,
) -> list[dict[str, Any]]:
    items = await store.asearch((client_id, EMOTION_FEEDBACK_NAMESPACE, thread_id), limit=1_000)
    return [item.value for item in items]


def load_emotion_feedback(
    store: BaseStore | None = None,
    client_id: str | None = None,
    thread_id: str | None = None,
) -> list[dict[str, Any]] | Coroutine[Any, Any, list[dict[str, Any]]]:
    """Load namespaced feedback, or retain the legacy runtime DB reader temporarily."""
    if store is None and client_id is None and thread_id is None:
        return RuntimeStore(RUNTIME_DB_PATH).load_json_records(EMOTION_FEEDBACK_NAMESPACE)
    if store is None or client_id is None or thread_id is None:
        raise TypeError("load_emotion_feedback requires store, client_id, and thread_id.")
    return _load_emotion_feedback(store, client_id, thread_id)


def _validated_feedback_record(record: dict[str, Any]) -> dict[str, Any]:
    feedback = str(record.get("feedback", "")).strip()
    if feedback not in ALLOWED_EMOTION_FEEDBACK:
        raise ValueError("Invalid emotion feedback.")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message_id": record.get("message_id", ""),
        "turn_count": record.get("turn_count"),
        "feedback": feedback,
        "predicted_emotion": record.get("predicted_emotion", ""),
        "corrected_emotion": record.get("corrected_emotion", ""),
    }


async def _append_emotion_feedback(
    store: BaseStore,
    client_id: str,
    thread_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    output = _validated_feedback_record(record)
    await store.aput((client_id, EMOTION_FEEDBACK_NAMESPACE, thread_id), str(uuid4()), output)
    return output


def append_emotion_feedback(
    store_or_record: BaseStore | dict[str, Any],
    client_id: str | None = None,
    thread_id: str | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any] | Coroutine[Any, Any, dict[str, Any]]:
    """Append namespaced feedback, or retain the legacy runtime DB writer temporarily."""
    if isinstance(store_or_record, dict) and client_id is None and thread_id is None and record is None:
        output = _validated_feedback_record(store_or_record)
        with _FEEDBACK_LOCK:
            RuntimeStore(RUNTIME_DB_PATH).append_json_record(EMOTION_FEEDBACK_NAMESPACE, output)
            return output
    if client_id is None or thread_id is None or record is None:
        raise TypeError("append_emotion_feedback requires store, client_id, thread_id, and record.")
    return _append_emotion_feedback(store_or_record, client_id, thread_id, record)
