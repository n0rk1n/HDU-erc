from chatbot.emotion_state import (
    EmotionState,
    emotion_state_from_output,
    format_emotion_state_context,
    timeline_from_records,
)


def test_emotion_state_normalizes_fields():
    state = EmotionState.from_mapping({
        "primary_emotion": "Anxious",
        "confidence": 1.4,
        "secondary_emotions": ["sad", "unknown", "anxious"],
        "evidence": " User is worried. ",
        "reply_strategy": " Be calm. ",
        "trajectory_note": " hopeful -> anxious ",
        "safety_level": "supportive",
    })

    assert state.primary_emotion == "anxious"
    assert state.confidence == 1.0
    assert state.secondary_emotions == ["sad"]
    assert state.evidence == "User is worried."
    assert state.reply_strategy == "Be calm."
    assert state.trajectory_note == "hopeful -> anxious"
    assert state.safety_level == "supportive"


def test_emotion_state_from_json_output():
    output = """
    {
      "primary_emotion": "joyful",
      "confidence": 0.82,
      "secondary_emotions": ["excited"],
      "evidence": "The user sounds happy.",
      "reply_strategy": "Share the positive energy.",
      "trajectory_note": "",
      "safety_level": "normal"
    }
    """

    state = emotion_state_from_output(output)

    assert state is not None
    assert state.primary_emotion == "joyful"
    assert state.confidence == 0.82
    assert state.secondary_emotions == ["excited"]


def test_emotion_state_rejects_unknown_primary_label():
    assert emotion_state_from_output('{"primary_emotion": "confused"}') is None


def test_format_emotion_state_context_is_concise():
    state = EmotionState(
        primary_emotion="anxious",
        confidence=0.78,
        secondary_emotions=["apprehensive", "sad"],
        evidence="The user expresses uncertainty.",
        reply_strategy="Use a calm tone.",
        trajectory_note="hopeful -> anxious",
        safety_level="normal",
    )

    context = format_emotion_state_context(state)

    assert "Current Emotion:" in context
    assert "- primary: anxious" in context
    assert "- confidence: 0.78" in context
    assert "- secondary: apprehensive, sad" in context
    assert "- evidence: The user expresses uncertainty." in context
    assert "- reply strategy: Use a calm tone." in context
    assert "safety guidance" not in context


def test_timeline_from_records_uses_successful_state_records():
    records = [
        {"success": False, "emotion": "", "state": {}},
        {
            "timestamp": "2026-06-20T10:00:00+08:00",
            "turn_count": 2,
            "success": True,
            "emotion": "sad",
            "state": {
                "primary_emotion": "sad",
                "confidence": 0.7,
                "secondary_emotions": ["lonely"],
                "evidence": "The user feels isolated.",
                "reply_strategy": "Validate the feeling.",
                "trajectory_note": "",
                "safety_level": "normal",
            },
        },
    ]

    timeline = timeline_from_records(records, limit=5)

    assert timeline == [{
        "timestamp": "2026-06-20T10:00:00+08:00",
        "turn_count": 2,
        "primary_emotion": "sad",
        "confidence": 0.7,
        "secondary_emotions": ["lonely"],
        "evidence": "The user feels isolated.",
        "reply_strategy": "Validate the feeling.",
        "trajectory_note": "",
        "safety_level": "normal",
    }]


def test_timeline_from_records_falls_back_to_legacy_emotion_record():
    records = [{
        "timestamp": "2026-06-20T11:00:00+08:00",
        "turn_count": 3,
        "success": True,
        "emotion": "joyful",
    }]

    timeline = timeline_from_records(records)

    assert timeline == [{
        "timestamp": "2026-06-20T11:00:00+08:00",
        "turn_count": 3,
        "primary_emotion": "joyful",
        "confidence": 0.0,
        "secondary_emotions": [],
        "evidence": "",
        "reply_strategy": "",
        "trajectory_note": "",
        "safety_level": "normal",
    }]


def test_timeline_from_records_returns_empty_for_non_positive_limit():
    records = [{
        "timestamp": "2026-06-20T11:00:00+08:00",
        "turn_count": 3,
        "success": True,
        "state": {"primary_emotion": "joyful"},
    }]

    assert timeline_from_records(records, limit=0) == []
    assert timeline_from_records(records, limit=-1) == []
