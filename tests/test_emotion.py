import json

from chatbot.emotion import (
    append_analysis_record,
    build_emotion_prompt,
    load_analysis_records,
    parse_emotion_output,
)


def test_build_emotion_prompt_uses_recent_history_and_current_input():
    records = [
        {"role": "human", "content": "first question"},
        {"role": "ai", "content": "first answer"},
        {"role": "human", "content": "second question"},
        {"role": "ai", "content": "second answer"},
    ]

    prompt = build_emotion_prompt(
        records,
        "current question",
        previous_emotion="anxious",
        max_turns=1,
    )

    assert "Emotion labels:" in prompt
    assert "Response Format: Emotion: [a single inferred emotion]" in prompt
    assert "More likely emotion label: anxious" in prompt
    assert "Dialogue context:" in prompt
    assert "first question" not in prompt
    assert "second question</s>second answer</s>current question" in prompt


def test_parse_emotion_output_accepts_known_label():
    assert parse_emotion_output("Emotion: anxious") == "anxious"


def test_parse_emotion_output_accepts_extra_text():
    output = "Emotion: joyful\nThe user sounds upbeat."

    assert parse_emotion_output(output) == "joyful"


def test_parse_emotion_output_rejects_unknown_label():
    assert parse_emotion_output("Emotion: confused") is None


def test_parse_emotion_output_rejects_missing_format():
    assert parse_emotion_output("The emotion is anxious.") is None


def test_load_analysis_records_missing_file(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert load_analysis_records() == []


def test_load_analysis_records_corrupted_file(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text("not json")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert load_analysis_records() == []


def test_append_analysis_record_creates_file(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    append_analysis_record({
        "turn_count": 5,
        "input": "prompt",
        "output": "Emotion: anxious",
        "emotion": "anxious",
        "success": True,
        "error": "",
    })

    data = json.loads(analysis_file.read_text())
    assert len(data) == 1
    assert data[0]["turn_count"] == 5
    assert data[0]["input"] == "prompt"
    assert data[0]["output"] == "Emotion: anxious"
    assert data[0]["emotion"] == "anxious"
    assert data[0]["success"] is True
    assert "timestamp" in data[0]


def test_append_analysis_record_appends(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    append_analysis_record({"turn_count": 5})
    append_analysis_record({"turn_count": 10})

    data = json.loads(analysis_file.read_text())
    assert [record["turn_count"] for record in data] == [5, 10]
