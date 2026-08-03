"""Emotion-correctness feedback persistence."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from chatbot.core.runtime_store import DEFAULT_RUNTIME_DB_PATH, RuntimeStore

RUNTIME_DB_PATH = DEFAULT_RUNTIME_DB_PATH
EMOTION_FEEDBACK_NAMESPACE = "emotion_feedback"
ALLOWED_EMOTION_FEEDBACK = {"accurate", "too_positive", "too_negative", "wrong_emotion"}
_FEEDBACK_LOCK = threading.RLock()


def load_emotion_feedback() -> list[dict[str, Any]]:
    return RuntimeStore(RUNTIME_DB_PATH).load_json_records(EMOTION_FEEDBACK_NAMESPACE)


def append_emotion_feedback(record: dict[str, Any]) -> dict[str, Any]:
    feedback = str(record.get("feedback", "")).strip()
    if feedback not in ALLOWED_EMOTION_FEEDBACK:
        raise ValueError("Invalid emotion feedback.")
    output = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "message_id": record.get("message_id", ""),
        "turn_count": record.get("turn_count"),
        "feedback": feedback,
        "predicted_emotion": record.get("predicted_emotion", ""),
        "corrected_emotion": record.get("corrected_emotion", ""),
    }
    with _FEEDBACK_LOCK:
        RuntimeStore(RUNTIME_DB_PATH).append_json_record(
            EMOTION_FEEDBACK_NAMESPACE,
            output,
        )
        return output
