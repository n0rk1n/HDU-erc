from chatbot.graphs.nodes.risk import risk_precheck


def test_risk_precheck_stores_browser_safe_codes_without_emitting_matched_phrase(writer, runtime):
    """Catches deterministic risk matching leaking the user's crisis phrase to event output."""
    phrase = "我现在想自杀"

    update = risk_precheck({"input_message": phrase}, runtime, writer)

    assert update["risk"] == {
        "signals": ["crisis_term", "first_person_intent"],
        "force_emotion_analysis": True,
        "explicit_crisis": True,
    }
    assert phrase not in repr(writer.events)


def test_risk_precheck_keeps_routine_turn_normal(writer, runtime):
    """Catches routine messages being forced through the crisis analysis branch."""
    update = risk_precheck({"input_message": "今天天气不错"}, runtime, writer)

    assert update["risk"] == {
        "signals": [],
        "force_emotion_analysis": False,
        "explicit_crisis": False,
    }
