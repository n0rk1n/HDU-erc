"""Emotion analysis domain."""

from chatbot.emotion.analysis import (
    EmotionAnalysisResult,
    analyze_emotion_async,
    build_emotion_prompt,
    parse_emotion_output,
)
from chatbot.emotion.labels import EMOTION_LABELS, EMOTION_LABEL_SET
from chatbot.emotion.prompt_variants import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANT_NAMES,
    resolve_emotion_prompt_template,
)
from chatbot.emotion.safety import assess_safety, precheck_risk

__all__ = [
    "EMOTION_LABELS",
    "EMOTION_LABEL_SET",
    "EmotionAnalysisResult",
    "analyze_emotion_async",
    "build_emotion_prompt",
    "parse_emotion_output",
    "DEFAULT_PROMPT_VARIANT",
    "PROMPT_VARIANT_NAMES",
    "resolve_emotion_prompt_template",
    "assess_safety",
    "precheck_risk",
]
