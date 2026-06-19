import json

from chatbot.emotion_feedback import append_emotion_feedback, load_emotion_feedback


def test_append_emotion_feedback_creates_file(tmp_path, monkeypatch):
    feedback_file = tmp_path / "emotion_feedback.json"
    monkeypatch.setattr("chatbot.emotion_feedback.EMOTION_FEEDBACK_FILE", str(feedback_file))

    record = append_emotion_feedback({
        "message_id": "ai_1",
        "turn_count": 2,
        "feedback": "wrong_emotion",
        "predicted_emotion": "sad",
        "corrected_emotion": "anxious",
    })

    assert record["feedback"] == "wrong_emotion"
    assert "timestamp" in record
    data = json.loads(feedback_file.read_text())
    assert data[0]["corrected_emotion"] == "anxious"


def test_load_emotion_feedback_returns_empty_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("chatbot.emotion_feedback.EMOTION_FEEDBACK_FILE", str(tmp_path / "missing.json"))

    assert load_emotion_feedback() == []
