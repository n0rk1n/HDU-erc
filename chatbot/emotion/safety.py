"""Local safety policy for emotion-aware replies."""

from __future__ import annotations

import re

from chatbot.emotion.state import EmotionState
from chatbot.models import RiskAssessment, SafetyDecision

CRISIS_TERMS = (
    "kill myself",
    "suicide",
    "end my life",
    "take my own life",
    "hurt myself",
    "hurting myself",
    "自杀",
    "不想活",
    "想死",
    "去死",
    "活不下去",
    "活不下去了",
    "结束生命",
    "结束自己的生命",
    "了结自己",
    "轻生",
    "自残",
    "伤害自己",
    "割腕",
    "跳楼",
    "一了百了",
    "离开这个世界",
)
SUPPORTIVE_TERMS = ("hopeless", "can't go on", "崩溃", "绝望", "撑不住")
SUPPORTIVE_EMOTIONS = {"devastated", "terrified", "afraid", "sad", "lonely", "anxious"}

_QUOTED_TEXT = re.compile(r'“[^”]*”|‘[^’]*’|"[^"]*"|\'[^\']*\'')
_CLAUSE_BOUNDARY = re.compile(r"[,.!?，。！？;；\n]+")
_CHINESE_FIRST_PERSON = re.compile(r"我(?!们|的)")
_CHINESE_STRONG_ACTION = re.compile(
    r"想死|不想活|活不下去|自杀|轻生|结束自己的?生命|了结自己|"
    r"伤害自己|割腕|跳楼|一了百了|离开这个世界"
)
_CHINESE_INTENT_OR_ACTION = re.compile(
    r"想|要|准备|打算|决定|计划|马上|今晚|现在|正在|已经|"
    r"不想活|活不下去|割腕|跳楼|一了百了"
)
_GENERAL_DISCUSSION = re.compile(r"讨论|研究|预防|报道|新闻|文章|演示")
_REPORTED_SPEECH = re.compile(r"(?:他|她|他们|她们|有人|朋友|家人|同事).{0,8}(?:说|表示|提到)")
_CHINESE_NEGATED_ACTION = re.compile(
    r"(?:没有|从未|并不|不是|没|不).{0,8}"
    r"(?:想死|不想活|自杀|轻生|结束自己的?生命|了结自己|"
    r"伤害自己|割腕|跳楼|一了百了|离开这个世界)"
)
_ENGLISH_FIRST_PERSON_ACTION = re.compile(
    r"\b(?:i\s+(?:want|plan|intend|decided|am going|will|am about|am trying)\s+to\s+"
    r"(?:kill myself|commit suicide|end my life|take my own life|die|hurt myself)|"
    r"i(?:'m| am)\s+(?:suicidal|cutting myself|hurting myself))\b"
)
_ENGLISH_NEGATED_ACTION = re.compile(
    r"\b(?:i\s+(?:do not|don't|never|no longer)\s+(?:want|plan|intend)|"
    r"i(?:'m| am)\s+not)\b.{0,24}"
    r"(?:kill myself|end my life|take my own life|die|suicidal|hurt myself)"
)


def precheck_risk(message: str) -> RiskAssessment:
    """Return deterministic browser-safe risk codes before model analysis."""
    text = message.lower()
    has_crisis_term = any(term in text for term in CRISIS_TERMS)
    has_distress = any(term in text for term in SUPPORTIVE_TERMS)
    explicit_crisis = has_crisis_term and _has_explicit_first_person_action(text)

    signals = []
    if has_crisis_term:
        signals.append("crisis_term")
    if has_distress:
        signals.append("severe_distress")
    if explicit_crisis:
        signals.append("first_person_intent")
    return {
        "signals": signals,
        "force_emotion_analysis": has_crisis_term or has_distress,
        "explicit_crisis": explicit_crisis,
    }


def _has_explicit_first_person_action(text: str) -> bool:
    unquoted = _QUOTED_TEXT.sub("", text)
    for clause in _CLAUSE_BOUNDARY.split(unquoted):
        if _ENGLISH_NEGATED_ACTION.search(clause):
            continue
        if _ENGLISH_FIRST_PERSON_ACTION.search(clause):
            return True
        if not _CHINESE_FIRST_PERSON.search(clause):
            continue
        if (
            _GENERAL_DISCUSSION.search(clause)
            or _REPORTED_SPEECH.search(clause)
            or _CHINESE_NEGATED_ACTION.search(clause)
        ):
            continue
        if _CHINESE_STRONG_ACTION.search(clause) and _CHINESE_INTENT_OR_ACTION.search(clause):
            return True
    return False


def assess_safety(
    message: str,
    state: EmotionState | None,
    *,
    risk: RiskAssessment | None = None,
) -> SafetyDecision:
    text = message.lower()
    risk = risk if risk is not None else precheck_risk(message)
    if risk["explicit_crisis"]:
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
