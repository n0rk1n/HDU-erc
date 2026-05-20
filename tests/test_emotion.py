import json
from pathlib import Path

import chatbot.emotion as emotion
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


def test_default_emotion_analysis_file_is_project_data_file():
    path = Path(emotion.EMOTION_ANALYSIS_FILE)

    assert path.is_absolute()
    assert path.name == "emotion_analysis.json"
    assert path.parent.name == "data"
    assert path.parent.parent == Path(__file__).resolve().parents[1]


def test_load_latest_successful_emotion_returns_latest_success(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text(json.dumps([
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "anxious",
            "success": True,
        },
        {
            "timestamp": "2026-05-19T18:10:00+08:00",
            "turn_count": 4,
            "emotion": "",
            "success": False,
            "error": "Failed to parse a known emotion label.",
        },
        {
            "timestamp": "2026-05-19T18:20:00+08:00",
            "turn_count": 5,
            "emotion": "sad",
            "success": True,
        },
    ]), encoding="utf-8")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "sad",
        "timestamp": "2026-05-19T18:20:00+08:00",
        "turn_count": 5,
    }


def test_load_latest_successful_emotion_skips_trailing_failures(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text(json.dumps([
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "anxious",
            "success": True,
        },
        {
            "timestamp": "2026-05-19T18:10:00+08:00",
            "turn_count": 4,
            "emotion": "",
            "success": False,
        },
    ]), encoding="utf-8")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "anxious",
        "timestamp": "2026-05-19T18:00:00+08:00",
        "turn_count": 3,
    }


def test_load_latest_successful_emotion_skips_malformed_trailing_records(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text(json.dumps([
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "anxious",
            "success": True,
        },
        {
            "timestamp": "2026-05-19T18:10:00+08:00",
            "turn_count": 4,
            "emotion": "sad",
            "success": "false",
        },
        {
            "timestamp": "2026-05-19T18:20:00+08:00",
            "turn_count": 5,
            "emotion": None,
            "success": True,
        },
    ]), encoding="utf-8")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "anxious",
        "timestamp": "2026-05-19T18:00:00+08:00",
        "turn_count": 3,
    }


def test_load_latest_successful_emotion_returns_none_without_success(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text(json.dumps([
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "",
            "success": False,
        }
    ]), encoding="utf-8")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() is None


def test_load_latest_successful_emotion_returns_none_for_missing_file(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() is None
