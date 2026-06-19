from chatbot.emotion_state import EmotionState
from chatbot.safety import assess_safety


def test_assess_safety_returns_normal_for_ordinary_message():
    result = assess_safety("I am preparing slides.", None)

    assert result["level"] == "normal"
    assert result["guidance"] == ""


def test_assess_safety_returns_supportive_for_distress():
    state = EmotionState(primary_emotion="devastated", confidence=0.9)

    result = assess_safety("I feel completely hopeless tonight.", state)

    assert result["level"] == "supportive"
    assert "supportive" in result["guidance"].lower()


def test_assess_safety_returns_crisis_for_self_harm_language():
    result = assess_safety("I want to kill myself.", None)

    assert result["level"] == "crisis"
    assert "immediate" in result["guidance"].lower()
