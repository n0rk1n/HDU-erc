import pytest

from chatbot.emotion import assess_safety, precheck_risk
from chatbot.emotion.state import EmotionState


@pytest.mark.parametrize(
    ("message", "signals", "force", "explicit"),
    [
        (
            "我今晚准备结束自己的生命。",
            ["crisis_term", "first_person_intent"],
            True,
            True,
        ),
        (
            "I want to kill myself.",
            ["crisis_term", "first_person_intent"],
            True,
            True,
        ),
        (
            "I plan to commit suicide tonight.",
            ["crisis_term", "first_person_intent"],
            True,
            True,
        ),
        (
            "I plan to take my own life tonight.",
            ["crisis_term", "first_person_intent"],
            True,
            True,
        ),
        (
            "I'm hurting myself right now.",
            ["crisis_term", "first_person_intent"],
            True,
            True,
        ),
        (
            "我现在正在割腕。",
            ["crisis_term", "first_person_intent"],
            True,
            True,
        ),
        (
            "我真的很绝望，感觉撑不住。",
            ["severe_distress"],
            True,
            False,
        ),
        ("新闻里有人说想死。", ["crisis_term"], True, False),
        ('他说：“我想自杀。”', ["crisis_term"], True, False),
        ("我的朋友想自杀。", ["crisis_term"], True, False),
        ("我们在讨论如何预防自杀。", ["crisis_term"], True, False),
        ("An article discusses the phrase kill myself.", ["crisis_term"], True, False),
        ("我没有想过自杀，只是在了解相关知识。", ["crisis_term"], True, False),
        ("I do not want to kill myself.", ["crisis_term"], True, False),
        ("我在准备明天的演示。", [], False, False),
    ],
)
def test_precheck_risk_distinguishes_context(
    message,
    signals,
    force,
    explicit,
):
    result = precheck_risk(message)

    assert result == {
        "signals": signals,
        "force_emotion_analysis": force,
        "explicit_crisis": explicit,
    }


@pytest.mark.parametrize(
    "message",
    [
        "我今晚准备去死。",
        "我决定结束生命。",
        "我打算自残。",
        "我不会告诉任何人就自杀。",
        "I'm going to kill myself tonight.",
    ],
)
def test_precheck_risk_recognizes_explicit_first_person_plans_and_actions(message):
    assert precheck_risk(message) == {
        "signals": ["crisis_term", "first_person_intent"],
        "force_emotion_analysis": True,
        "explicit_crisis": True,
    }


@pytest.mark.parametrize(
    "message",
    [
        "He said I want to kill myself.",
        "If I want to kill myself, where can I get help?",
    ],
)
def test_precheck_risk_keeps_reported_or_conditional_language_contextual(message):
    assert precheck_risk(message) == {
        "signals": ["crisis_term"],
        "force_emotion_analysis": True,
        "explicit_crisis": False,
    }


def test_assess_safety_returns_normal_for_ordinary_message():
    result = assess_safety("I am preparing slides.", None)

    assert result["level"] == "normal"
    assert result["guidance"] == ""


def test_explicit_crisis_cannot_be_downgraded_by_normal_emotion():
    risk = {
        "signals": ["first_person_intent"],
        "force_emotion_analysis": True,
        "explicit_crisis": True,
    }

    decision = assess_safety(
        "我今晚准备结束自己的生命。",
        EmotionState("calm"),
        risk=risk,
    )

    assert decision["level"] == "crisis"


def test_assess_safety_returns_supportive_for_distress():
    state = EmotionState(primary_emotion="devastated", confidence=0.9)

    result = assess_safety("I feel completely hopeless tonight.", state)

    assert result["level"] == "supportive"
    assert "supportive" in result["guidance"].lower()


def test_assess_safety_returns_crisis_for_self_harm_language():
    result = assess_safety("I want to kill myself.", None)

    assert result["level"] == "crisis"
    assert "immediate" in result["guidance"].lower()


def test_assess_safety_returns_crisis_for_common_chinese_self_harm_language():
    result = assess_safety("我真的活不下去了，想死。", None)

    assert result["level"] == "crisis"
    assert "immediate" in result["guidance"].lower()
