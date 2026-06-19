"""Local safety policy for emotion-aware replies."""

from __future__ import annotations

from chatbot.emotion_state import EmotionState

CRISIS_TERMS = ("kill myself", "suicide", "end my life", "自杀", "不想活")
SUPPORTIVE_TERMS = ("hopeless", "can't go on", "崩溃", "绝望", "撑不住")
SUPPORTIVE_EMOTIONS = {"devastated", "terrified", "afraid", "sad", "lonely", "anxious"}


def assess_safety(message: str, state: EmotionState | None) -> dict[str, str]:
    text = message.lower()
    if any(term in text for term in CRISIS_TERMS):
        return {
            "level": "crisis",
            "guidance": (
                "Use immediate supportive language, avoid diagnosis, and encourage the user "
                "to contact trusted people or local emergency/professional support now."
            ),
        }
    if any(term in text for term in SUPPORTIVE_TERMS):
        return {
            "level": "supportive",
            "guidance": "Use supportive validation before practical next steps.",
        }
    if state and state.primary_emotion in SUPPORTIVE_EMOTIONS and state.confidence >= 0.85:
        return {
            "level": "supportive",
            "guidance": "Use supportive validation before practical next steps.",
        }
    return {"level": "normal", "guidance": ""}
