"""Emotion-correctness feedback persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EMOTION_FEEDBACK_FILE = str(DATA_DIR / "records" / "emotion_feedback.json")
ALLOWED_EMOTION_FEEDBACK = {"accurate", "too_positive", "too_negative", "wrong_emotion"}


def load_emotion_feedback() -> list[dict[str, Any]]:
    path = Path(EMOTION_FEEDBACK_FILE)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


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
    records = load_emotion_feedback()
    records.append(output)
    path = Path(EMOTION_FEEDBACK_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
