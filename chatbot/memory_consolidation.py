"""Periodic long-term memory consolidation helpers."""

import os
import re
from dataclasses import dataclass

from chatbot.emotion_state import EmotionState
from chatbot.memory import MemoryCandidate, MemoryRuntimeConfig

DEFAULT_CONSOLIDATION_INTERVAL = 5
DEFAULT_CONSOLIDATION_WINDOW = 12
MAX_CONSOLIDATED_CANDIDATES = 3
SUPPORTED_CONSOLIDATION_MODES = {"rules"}
STRESSOR_LABELS = ("项目压力", "工作压力", "学习压力", "家庭压力")


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


def consolidation_due(
    config: MemoryConsolidationConfig,
    *,
    turn_count: int,
    last_turn_count: int,
) -> bool:
    if not config.enabled:
        return False
    if turn_count <= 0:
        return False
    return turn_count - last_turn_count >= config.interval


def recent_consolidation_window(
    records: list[dict],
    *,
    window: int,
    last_message_id: str | None,
) -> list[dict]:
    filtered = [
        record
        for record in records
        if record.get("role") in {"human", "ai"}
        and str(record.get("content", "")).strip()
    ]
    if last_message_id:
        for index, record in enumerate(filtered):
            if record.get("id") == last_message_id:
                filtered = filtered[index + 1:]
                break
    return filtered[-max(1, window):]


def extract_consolidated_memory_candidates(records: list[dict]) -> list[MemoryCandidate]:
    human_texts = [
        str(record.get("content", "")).strip()
        for record in records
        if record.get("role") == "human" and str(record.get("content", "")).strip()
    ]
    candidates: list[MemoryCandidate] = []

    if any(_mentions_listening_preference(text) for text in human_texts):
        candidates.append(
            MemoryCandidate(
                content="用户希望难受时先被倾听，不要被急着建议。",
                category="preference",
                confidence=0.85,
            )
        )

    joined = "\n".join(human_texts)
    if _mentions_no_cheer_up(joined):
        candidates.append(
            MemoryCandidate(
                content="用户要求不要用劝用户想开点的方式回应。",
                category="boundary",
                confidence=0.9,
            )
        )

    stressor = _repeated_stressor(human_texts)
    if stressor:
        candidates.append(
            MemoryCandidate(
                content=f"用户在最近对话中多次提到{stressor}。",
                category="profile",
                confidence=0.75,
            )
        )

    return _dedupe_candidates(candidates)[:MAX_CONSOLIDATED_CANDIDATES]


def _mentions_listening_preference(text: str) -> bool:
    listening_pattern = r"(只是想被听见|只想被听见|只是想被倾听|只想被倾听|I just need you to listen)"
    advice_boundary_pattern = r"(不要急着给建议|不急着给建议|不要建议|不要[^。！？\n]*建议)"
    return bool(
        re.search(listening_pattern, text, re.IGNORECASE)
        and re.search(advice_boundary_pattern, text)
    )


def _mentions_no_cheer_up(text: str) -> bool:
    return bool(re.search(r"(不要|别)劝我想开点", text))


def _repeated_stressor(human_texts: list[str]) -> str | None:
    for label in STRESSOR_LABELS:
        mentions = sum(1 for text in human_texts if re.search(re.escape(label), text))
        if mentions >= 2:
            return label
    return None


def _dedupe_candidates(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    deduped: list[MemoryCandidate] = []
    seen = set()
    for candidate in candidates:
        if candidate.content in seen:
            continue
        seen.add(candidate.content)
        deduped.append(candidate)
    return deduped


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
