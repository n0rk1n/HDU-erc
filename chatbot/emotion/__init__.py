"""Emotion analysis domain."""

from chatbot.emotion.analysis import (
    EmotionAnalysisResult,
    analyze_emotion,
    append_analysis_record,
    build_emotion_prompt,
    load_analysis_records,
    load_latest_successful_emotion,
    parse_emotion_output,
    successful_emotion_snapshot,
)
from chatbot.emotion.labels import EMOTION_LABELS, EMOTION_LABEL_SET
from chatbot.emotion.prompt_variants import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANT_NAMES,
    resolve_emotion_prompt_template,
)

__all__ = [
    "EMOTION_LABELS",
    "EMOTION_LABEL_SET",
    "EmotionAnalysisResult",
    "analyze_emotion",
    "append_analysis_record",
    "build_emotion_prompt",
    "load_analysis_records",
    "load_latest_successful_emotion",
    "parse_emotion_output",
    "successful_emotion_snapshot",
    "DEFAULT_PROMPT_VARIANT",
    "PROMPT_VARIANT_NAMES",
    "resolve_emotion_prompt_template",
]
