import pytest

from chatbot.graphs.routing import (
    route_by_safety,
    route_emotion,
    route_operation,
    should_analyze_emotion,
)


@pytest.mark.parametrize(
    ("turn_count", "last_turn", "force", "expected"),
    [(1, 0, False, True), (3, 1, False, False), (6, 1, False, True), (3, 1, True, True)],
)
def test_should_analyze_emotion_uses_interval_or_explicit_force(turn_count, last_turn, force, expected):
    """Catches emotion analysis being skipped at its interval or when a risk signal forces it."""
    state = {
        "turn_count": turn_count,
        "last_emotion_analysis_turn": last_turn,
        "risk": {"signals": [], "force_emotion_analysis": force, "explicit_crisis": False},
    }
    assert should_analyze_emotion(state, emotion_interval=5) is expected


def test_route_by_safety_uses_dedicated_crisis_node():
    """Catches crisis safety decisions falling through to the ordinary reply generator."""
    assert route_by_safety({"safety_state": {"level": "crisis"}}) == "generate_crisis_reply"
    assert route_by_safety({"safety_state": {"level": "supportive"}}) == "generate_reply"


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("turn", "turn"), ("regenerate", "regenerate"), ("onboard", "onboard"), ("unknown", "invalid")],
)
def test_route_operation_rejects_unknown_operation(operation, expected):
    """Catches unsupported operations reaching a conversation-processing node."""
    assert route_operation({"operation": operation}) == expected


def test_route_emotion_reuses_state_when_interval_has_not_elapsed():
    """Catches routine turns triggering an unnecessary emotion-model invocation."""
    state = {
        "turn_count": 3,
        "last_emotion_analysis_turn": 1,
        "risk": {"signals": [], "force_emotion_analysis": False, "explicit_crisis": False},
    }
    assert route_emotion(state, emotion_interval=5) == "reuse_emotion"
