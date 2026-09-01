"""Local safety policy for emotion-aware replies."""

from __future__ import annotations

import re

from chatbot.emotion.state import EmotionState
from chatbot.models import RiskAssessment, SafetyDecision

_ENGLISH_ACTION_TERMS = (
    "kill myself",
    "commit suicide",
    "end my life",
    "take my own life",
    "hurt myself",
    "hurting myself",
    "cut myself",
    "cutting myself",
    "suicidal",
)
_CHINESE_ACTION_TERMS = (
    "自杀",
    "去死",
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
_CHINESE_DISTRESS_TERMS = ("不想活", "想死", "活不下去", "活不下去了")
CRISIS_TERMS = (
    *_ENGLISH_ACTION_TERMS,
    "suicide",
    *_CHINESE_ACTION_TERMS,
    *_CHINESE_DISTRESS_TERMS,
)
SUPPORTIVE_TERMS = ("hopeless", "can't go on", "崩溃", "绝望", "撑不住")
SUPPORTIVE_EMOTIONS = {"devastated", "terrified", "afraid", "sad", "lonely", "anxious"}
_SAFETY_RANK = {"normal": 0, "supportive": 1, "crisis": 2}
_SAFETY_GUIDANCE = {
    "normal": "",
    "supportive": "Use supportive validation before practical next steps.",
    "crisis": (
        "Use immediate supportive language, avoid diagnosis, and encourage the user "
        "to contact trusted people or local emergency/professional support now."
    ),
}

_QUOTED_TEXT = re.compile(r'“[^”]*”|‘[^’]*’|"[^"]*"|\'[^\']*\'')
_CLAUSE_BOUNDARY = re.compile(r"[,.!?，。！？;；\n]+")
_CHINESE_FIRST_PERSON = re.compile(r"我(?!们|的)")
_CHINESE_ACTION_PATTERN = "|".join(
    re.escape(term) for term in sorted(_CHINESE_ACTION_TERMS, key=len, reverse=True)
)
_ENGLISH_ACTION_PATTERN = "|".join(
    re.escape(term) for term in sorted(_ENGLISH_ACTION_TERMS, key=len, reverse=True)
)
_CHINESE_DIRECT_DISTRESS = re.compile(
    "|".join(re.escape(term) for term in _CHINESE_DISTRESS_TERMS)
)
_CHINESE_EXPLICIT_ACTION = re.compile(
    rf"(?:想(?:要)?|要|准备|打算|决定|计划|马上|今晚|现在|正在|已经|就)"
    rf"[^，。！？\n]{{0,8}}(?:{_CHINESE_ACTION_PATTERN})"
)
_GENERAL_DISCUSSION = re.compile(r"讨论|研究|预防|报道|新闻|文章|演示")
_REPORTED_SPEECH = re.compile(
    r"(?:他|她|他们|她们|有人|朋友|家人|同事).{0,8}(?:说|表示|提到)|"
    r"\b(?:he|she|they|someone|my friend|my family|a friend)\s+"
    r"(?:said|says|reported|mentioned)\b"
)
_CONDITIONAL_CONTEXT = re.compile(
    r"^\s*(?:(?:if|suppose|assuming)\b|如果|假如|假设|万一)"
)
_CHINESE_NEGATED_ACTION = re.compile(
    rf"(?:不想|不打算|不准备|不计划|不会|并不想|"
    rf"没有(?:想过|打算|计划|准备|决定|要|去)?|"
    rf"没(?:想过|打算|计划|准备|决定|要|去)?|"
    rf"从未(?:想过|打算|计划|准备|决定|要|去)?)"
    rf"(?:要|去|再)?(?:{_CHINESE_ACTION_PATTERN})"
)
_ENGLISH_FIRST_PERSON_ACTION = re.compile(
    rf"\b(?:i\s+(?:want|plan|intend|decided|am going|am about|am trying)\s+to\s+"
    rf"(?:{_ENGLISH_ACTION_PATTERN})|i\s+will\s+(?:{_ENGLISH_ACTION_PATTERN})|"
    rf"i(?:'m| am)\s+(?:going|planning|about|trying)\s+to\s+(?:{_ENGLISH_ACTION_PATTERN})|"
    rf"i(?:'m| am)\s+(?:suicidal|cutting myself|hurting myself))\b"
)
_ENGLISH_NEGATED_ACTION = re.compile(
    rf"\b(?:i\s+(?:do not|don't|never|no longer)\s+(?:want|plan|intend)\s+to|"
    rf"i(?:'m| am)\s+not\s+(?:going|planning|about|trying)\s+to|i\s+will\s+not)"
    rf"\s+(?:{_ENGLISH_ACTION_PATTERN})\b|\bi(?:'m| am)\s+not\s+suicidal\b"
)
_CHINESE_CURRENT_POISONING = re.compile(
    r"我(?!们|的).{0,10}(?:刚刚|刚才|已经|现在|正在).{0,8}"
    r"(?:吞(?:了|下)?(?:很多|大量)?药|吃(?:了|下)?(?:很多|大量|过量)?药|"
    r"过量服药|服毒|喝(?:了)?农药)"
)
_ENGLISH_CURRENT_POISONING = re.compile(
    r"\bi\s+(?:(?:just|already)\s+(?:took|swallowed|drank)|"
    r"have\s+(?:just|already)\s+(?:taken|swallowed|drunk)|"
    r"am\s+(?:taking|swallowing|drinking))\s+"
    r"(?:an?\s+overdose|too\s+many\s+pills?|pills?|poison|pesticide)\b"
)
_CHINESE_DANGEROUS_LOCATION = re.compile(
    r"我(?!们|的).{0,10}(?:现在|已经|就)?(?:正在|站在|待在|坐在|走在?)"
    r"(?:楼顶|天台|桥边|桥上|铁轨|轨道).{0,18}"
    r"(?:不想活|想死|自杀|跳下去|结束生命|一了百了)"
)
_ENGLISH_DANGEROUS_LOCATION = re.compile(
    r"\bi(?:'m| am)\s+(?:now\s+)?(?:standing|sitting|waiting|walking)\s+"
    r"(?:on|at|by|beside|near)\s+(?:a|the)?\s*"
    r"(?:rooftop|roof|bridge|railroad tracks?|train tracks?)\b.{0,80}"
    r"\b(?:want to die|kill myself|end my life|jump|suicidal)\b"
)


def precheck_risk(message: str) -> RiskAssessment:
    """Return deterministic browser-safe risk codes before model analysis."""
    text = message.lower()
    contextual_current_danger = _has_contextual_current_danger(text)
    has_crisis_term = any(term in text for term in CRISIS_TERMS) or contextual_current_danger
    has_distress = any(term in text for term in SUPPORTIVE_TERMS)
    explicit_crisis = contextual_current_danger or (
        has_crisis_term and _has_explicit_first_person_action(text)
    )

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
        if _REPORTED_SPEECH.search(clause) or _CONDITIONAL_CONTEXT.search(clause):
            continue
        if _ENGLISH_NEGATED_ACTION.search(clause):
            continue
        if _ENGLISH_FIRST_PERSON_ACTION.search(clause):
            return True
        if not _CHINESE_FIRST_PERSON.search(clause):
            continue
        if (
            _GENERAL_DISCUSSION.search(clause)
            or _CHINESE_NEGATED_ACTION.search(clause)
        ):
            continue
        if _CHINESE_DIRECT_DISTRESS.search(clause) or _CHINESE_EXPLICIT_ACTION.search(clause):
            return True
    return False


def _has_contextual_current_danger(text: str) -> bool:
    unquoted = _QUOTED_TEXT.sub("", text)
    if not (
        _REPORTED_SPEECH.search(unquoted)
        or _CONDITIONAL_CONTEXT.search(unquoted)
        or _GENERAL_DISCUSSION.search(unquoted)
    ) and any(
        pattern.search(unquoted)
        for pattern in (
            _CHINESE_CURRENT_POISONING,
            _ENGLISH_CURRENT_POISONING,
            _CHINESE_DANGEROUS_LOCATION,
            _ENGLISH_DANGEROUS_LOCATION,
        )
    ):
        return True
    for clause in _CLAUSE_BOUNDARY.split(unquoted):
        if _REPORTED_SPEECH.search(clause) or _CONDITIONAL_CONTEXT.search(clause):
            continue
        if _GENERAL_DISCUSSION.search(clause):
            continue
        if any(
            pattern.search(clause)
            for pattern in (
                _CHINESE_CURRENT_POISONING,
                _ENGLISH_CURRENT_POISONING,
                _CHINESE_DANGEROUS_LOCATION,
                _ENGLISH_DANGEROUS_LOCATION,
            )
        ):
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
    local_level = "normal"
    if risk["explicit_crisis"]:
        local_level = "crisis"
    elif any(term in text for term in SUPPORTIVE_TERMS):
        local_level = "supportive"
    elif state and state.primary_emotion in SUPPORTIVE_EMOTIONS and state.confidence >= 0.85:
        local_level = "supportive"

    model_level = state.safety_level if state is not None else "normal"
    final_level = max(
        (local_level, model_level),
        key=_SAFETY_RANK.__getitem__,
    )
    return {
        "level": final_level,
        "guidance": _SAFETY_GUIDANCE[final_level],
    }
