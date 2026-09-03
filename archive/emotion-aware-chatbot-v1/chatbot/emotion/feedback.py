"""Emotion-correctness feedback persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langgraph.store.base import BaseStore

EMOTION_FEEDBACK_NAMESPACE = "emotion_feedback"
ALLOWED_EMOTION_FEEDBACK = {"accurate", "too_positive", "too_negative", "wrong_emotion"}


async def load_emotion_feedback(
    store: BaseStore,
    client_id: str,
    thread_id: str,
) -> list[dict[str, Any]]:
    items = await store.asearch((client_id, EMOTION_FEEDBACK_NAMESPACE, thread_id), limit=1_000)
    return [item.value for item in items]


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


async def append_emotion_feedback(
    store: BaseStore,
    client_id: str,
    thread_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    output = _validated_feedback_record(record)
    await store.aput((client_id, EMOTION_FEEDBACK_NAMESPACE, thread_id), str(uuid4()), output)
    return output
