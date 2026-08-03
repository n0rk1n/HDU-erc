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
]
