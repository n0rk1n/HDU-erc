from chatbot.emotion_feedback import append_emotion_feedback, load_emotion_feedback


def test_append_emotion_feedback_creates_database_record(tmp_path, monkeypatch):
    runtime_db = tmp_path / "runtime.sqlite3"
    monkeypatch.setattr("chatbot.emotion_feedback.RUNTIME_DB_PATH", str(runtime_db))

    record = append_emotion_feedback({
        "message_id": "ai_1",
        "turn_count": 2,
        "feedback": "wrong_emotion",
        "predicted_emotion": "sad",
        "corrected_emotion": "anxious",
    })

    assert record["feedback"] == "wrong_emotion"
    assert "timestamp" in record
    data = load_emotion_feedback()
    assert data[0]["corrected_emotion"] == "anxious"


def test_load_emotion_feedback_returns_empty_for_missing_database(tmp_path, monkeypatch):
    monkeypatch.setattr("chatbot.emotion_feedback.RUNTIME_DB_PATH", str(tmp_path / "missing.sqlite3"))

    assert load_emotion_feedback() == []
