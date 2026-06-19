"""Structured emotion state for emotion-aware chat context."""

import json
import re
from dataclasses import dataclass
from typing import Any

from chatbot.emotion_labels import EMOTION_LABEL_SET

SAFETY_LEVELS = {"normal", "supportive", "crisis"}


@dataclass(frozen=True)
class EmotionState:
    primary_emotion: str
    confidence: float = 0.0
    secondary_emotions: list[str] | None = None
    evidence: str = ""
    reply_strategy: str = ""
    trajectory_note: str = ""
    safety_level: str = "normal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "secondary_emotions", list(self.secondary_emotions or []))

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EmotionState | None":
        primary_emotion = _normalize_label(value.get("primary_emotion") or value.get("Emotion") or value.get("emotion"))
        if primary_emotion not in EMOTION_LABEL_SET:
            return None

        return cls(
            primary_emotion=primary_emotion,
            confidence=_clamp_confidence(value.get("confidence", 0.0)),
            secondary_emotions=_normalize_secondary_emotions(value.get("secondary_emotions"), primary_emotion),
            evidence=_clean_text(value.get("evidence", "")),
            reply_strategy=_clean_text(value.get("reply_strategy", "")),
            trajectory_note=_clean_text(value.get("trajectory_note", "")),
            safety_level=_normalize_safety_level(value.get("safety_level", "normal")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion,
            "confidence": self.confidence,
            "secondary_emotions": list(self.secondary_emotions or []),
            "evidence": self.evidence,
            "reply_strategy": self.reply_strategy,
            "trajectory_note": self.trajectory_note,
            "safety_level": self.safety_level,
        }


def emotion_state_from_output(output: str) -> EmotionState | None:
    mapping = _parse_json_object(output)
    if mapping is None:
        mapping = _parse_structured_lines(output)
    if mapping is None:
        return None
    return EmotionState.from_mapping(mapping)


def format_emotion_state_context(state: EmotionState) -> str:
    lines = [
        "Current Emotion:",
        f"- primary: {state.primary_emotion}",
        f"- confidence: {state.confidence:.2f}",
    ]
    if state.secondary_emotions:
        lines.append(f"- secondary: {', '.join(state.secondary_emotions)}")
    if state.evidence:
        lines.append(f"- evidence: {state.evidence}")
    if state.reply_strategy:
        lines.append(f"- reply strategy: {state.reply_strategy}")
    if state.trajectory_note:
        lines.append(f"- trajectory: {state.trajectory_note}")
    if state.safety_level != "normal":
        lines.append(f"- safety guidance: {state.safety_level}")
    return "\n".join(lines)


def timeline_from_records(records: list[dict], limit: int = 10) -> list[dict[str, Any]]:
    timeline = []
    for record in records:
        if not isinstance(record, dict) or record.get("success") is not True:
            continue
        state_value = record.get("state")
        if not isinstance(state_value, dict):
            continue
        state = EmotionState.from_mapping(state_value)
        if state is None:
            continue
        timeline.append({
            "timestamp": record.get("timestamp", ""),
            "turn_count": record.get("turn_count", 0),
            **state.to_dict(),
        })
    return timeline[-max(0, limit):]


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, confidence))


def _normalize_secondary_emotions(value: Any, primary_emotion: str) -> list[str]:
    if isinstance(value, str):
        candidates = _parse_secondary_string(value)
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []

    normalized = []
    for candidate in candidates:
        emotion = _normalize_label(candidate)
        if emotion in EMOTION_LABEL_SET and emotion != primary_emotion and emotion not in normalized:
            normalized.append(emotion)
    return normalized


def _parse_secondary_string(value: str) -> list[Any]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return data
    return [part.strip() for part in stripped.split(",")]


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_safety_level(value: Any) -> str:
    safety_level = str(value or "").strip().lower()
    if safety_level not in SAFETY_LEVELS:
        return "normal"
    return safety_level


def _parse_json_object(output: str) -> dict[str, Any] | None:
    text = output.strip()
    if not text:
        return None
    data = _loads_object(text)
    if data is not None:
        return data

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    return _loads_object(match.group(0))


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_structured_lines(output: str) -> dict[str, Any] | None:
    mapping: dict[str, Any] = {}
    key_aliases = {
        "emotion": "primary_emotion",
        "primary_emotion": "primary_emotion",
        "confidence": "confidence",
        "secondary_emotions": "secondary_emotions",
        "evidence": "evidence",
        "reply_strategy": "reply_strategy",
        "trajectory_note": "trajectory_note",
        "safety_level": "safety_level",
    }
    for line in output.splitlines():
        match = re.match(r"\s*[-*]?\s*([A-Za-z_ ]+)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        raw_key, value = match.groups()
        key = raw_key.strip().lower().replace(" ", "_")
        alias = key_aliases.get(key)
        if alias is None:
            continue
        mapping[alias] = value
    return mapping or None
