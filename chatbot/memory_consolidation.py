"""Periodic long-term memory consolidation helpers."""

import os
from dataclasses import dataclass

from chatbot.emotion_state import EmotionState
from chatbot.memory import MemoryRuntimeConfig

DEFAULT_CONSOLIDATION_INTERVAL = 5
DEFAULT_CONSOLIDATION_WINDOW = 12
SUPPORTED_CONSOLIDATION_MODES = {"rules"}


@dataclass(frozen=True)
class MemoryConsolidationConfig:
    enabled: bool
    interval: int
    window: int
    mode: str


def load_memory_consolidation_config(
    memory_config: MemoryRuntimeConfig,
) -> MemoryConsolidationConfig:
    enabled = memory_config.enabled and _parse_bool(
        os.getenv("MEMORY_CONSOLIDATION_ENABLED"),
        default=True,
    )
    interval = _parse_positive_int(
        os.getenv("MEMORY_CONSOLIDATION_INTERVAL"),
        default=DEFAULT_CONSOLIDATION_INTERVAL,
    )
    window = _parse_positive_int(
        os.getenv("MEMORY_CONSOLIDATION_WINDOW"),
        default=DEFAULT_CONSOLIDATION_WINDOW,
    )
    mode = _clean(os.getenv("MEMORY_CONSOLIDATION_MODE")) or "rules"
    if mode not in SUPPORTED_CONSOLIDATION_MODES:
        mode = "rules"
    return MemoryConsolidationConfig(
        enabled=enabled,
        interval=interval,
        window=window,
        mode=mode,
    )


def build_memory_search_query(
    message: str,
    emotion_state: EmotionState | str | None,
    recent_emotions: list[str],
) -> str:
    parts = [message.strip()]
    current_emotion = _emotion_name(emotion_state)
    if current_emotion:
        parts.append(f"Current emotion: {current_emotion}")
    deduped_recent = _dedupe_preserving_order(recent_emotions)
    if deduped_recent:
        parts.append(f"Recent emotions: {', '.join(deduped_recent)}")
    return "\n".join(part for part in parts if part)


def _emotion_name(emotion_state: EmotionState | str | None) -> str:
    if emotion_state is None:
        return ""
    if isinstance(emotion_state, EmotionState):
        return emotion_state.primary_emotion.strip()
    return str(emotion_state).strip()


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_bool(value: str | None, *, default: bool) -> bool:
    value = _clean(value)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _parse_positive_int(value: str | None, *, default: int) -> int:
    value = _clean(value)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed
