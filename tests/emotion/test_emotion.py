from pathlib import Path

import chatbot.emotion.analysis as emotion
from chatbot.core.runtime_store import RuntimeStore
from chatbot.emotion import (
    append_analysis_record,
    build_emotion_prompt,
    load_analysis_records,
    parse_emotion_output,
)


def _set_runtime_db(tmp_path, monkeypatch):
    runtime_db = tmp_path / "runtime.sqlite3"
    monkeypatch.setattr("chatbot.emotion.analysis.RUNTIME_DB_PATH", str(runtime_db))
    return runtime_db


def _replace_analysis_records(db_path, records):
    RuntimeStore(str(db_path)).replace_json_records(
        emotion.EMOTION_ANALYSIS_NAMESPACE,
        records,
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
    assert "Response Format: Return exactly one JSON object" in prompt
    assert '"primary_emotion": "anxious"' in prompt
    assert '"reply_strategy": "brief guidance for the next chatbot reply"' in prompt
    assert "More likely emotion labels: anxious" in prompt
    assert "Dynamic EICL examples:" in prompt
    assert "True emotion label: anxious" in prompt
    assert "recent-emotion-prior" in prompt
    assert "Dialogue context:" in prompt
    assert "first question" not in prompt
    assert "second question</s>second answer</s>current question" in prompt


def test_build_emotion_prompt_accepts_multiple_likely_emotions():
    prompt = build_emotion_prompt(
        [],
        "I feel let down but still nervous about tomorrow.",
        previous_emotion="anxious",
        likely_emotions=["disappointed", "sad", "unknown", "anxious"],
    )

    assert "More likely emotion labels: anxious, disappointed, sad" in prompt
    assert "True emotion label: anxious" in prompt
    assert "True emotion label: disappointed" in prompt


def test_build_emotion_prompt_uses_previous_and_likely_emotions_for_dynamic_examples():
    prompt = build_emotion_prompt(
        [],
        "thank you",
        previous_emotion="angry",
        likely_emotions=["grateful"],
    )

    assert "More likely emotion labels: angry, grateful" in prompt
    assert "True emotion label: angry" in prompt
    assert "True emotion label: grateful" in prompt


def test_build_emotion_prompt_can_disable_dynamic_and_static_examples():
    prompt = build_emotion_prompt(
        [],
        "I feel nervous about tomorrow.",
        previous_emotion="anxious",
        example_mode="none",
    )

    assert "Dynamic EICL examples:" not in prompt
    assert "Labeled examples:" not in prompt
    assert "More likely emotion labels: anxious" in prompt


def test_build_emotion_prompt_can_disable_emotion_history():
    prompt = build_emotion_prompt(
        [],
        "thank you",
        previous_emotion="angry",
        likely_emotions=["grateful"],
        include_emotion_history=False,
    )

    assert "More likely emotion labels:" not in prompt
    assert "recent-emotion-prior" not in prompt


def test_build_emotion_prompt_supports_static_examples_mode():
    prompt = build_emotion_prompt(
        [],
        "thank you",
        previous_emotion="grateful",
        example_mode="static",
    )

    assert "Dynamic EICL examples:" not in prompt
    assert "Labeled examples:" in prompt
    assert "True emotion label: grateful" in prompt


def test_build_emotion_prompt_passes_prompt_variant_through():
    prompt = build_emotion_prompt(
        [],
        "thank you",
        prompt_variant="prompt_coarse_to_fine",
    )

    assert "First identify the broad emotion family internally" in prompt


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
    _set_runtime_db(tmp_path, monkeypatch)

    assert load_analysis_records() == []


def test_load_analysis_records_corrupted_file(tmp_path, monkeypatch):
    runtime_db = _set_runtime_db(tmp_path, monkeypatch)
    runtime_db.write_text("not sqlite")

    assert load_analysis_records() == []


def test_append_analysis_record_creates_database_record(tmp_path, monkeypatch):
    _set_runtime_db(tmp_path, monkeypatch)

    append_analysis_record({
        "turn_count": 5,
        "input": "prompt",
        "output": "Emotion: anxious",
        "emotion": "anxious",
        "success": True,
        "error": "",
    })

    data = load_analysis_records()
    assert len(data) == 1
    assert data[0]["turn_count"] == 5
    assert data[0]["input"] == "prompt"
    assert data[0]["output"] == "Emotion: anxious"
    assert data[0]["emotion"] == "anxious"
    assert data[0]["success"] is True
    assert "timestamp" in data[0]


def test_append_analysis_record_appends(tmp_path, monkeypatch):
    _set_runtime_db(tmp_path, monkeypatch)

    append_analysis_record({"turn_count": 5})
    append_analysis_record({"turn_count": 10})

    data = load_analysis_records()
    assert [record["turn_count"] for record in data] == [5, 10]


def test_default_emotion_analysis_file_is_project_data_file():
    path = Path(emotion.RUNTIME_DB_PATH)

    assert path.is_absolute()
    assert path.name == "runtime.sqlite3"
    assert path.parent.name == "records"
    assert path.parent.parent.name == "data"
    assert path.parent.parent.parent == Path(__file__).resolve().parents[2]


def test_load_analysis_records_ignores_legacy_data_file_when_database_missing(tmp_path, monkeypatch):
    runtime_db = tmp_path / "data" / "records" / "runtime.sqlite3"
    legacy_file = tmp_path / "data" / "emotion_analysis.json"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text('[{"emotion": "sad"}]')
    monkeypatch.setattr("chatbot.emotion.analysis.RUNTIME_DB_PATH", str(runtime_db))

    assert load_analysis_records() == []


def test_load_latest_successful_emotion_returns_latest_success(tmp_path, monkeypatch):
    runtime_db = _set_runtime_db(tmp_path, monkeypatch)
    _replace_analysis_records(runtime_db, [
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
    ])

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "sad",
        "timestamp": "2026-05-19T18:20:00+08:00",
        "turn_count": 5,
    }


def test_load_latest_successful_emotion_skips_trailing_failures(tmp_path, monkeypatch):
    runtime_db = _set_runtime_db(tmp_path, monkeypatch)
    _replace_analysis_records(runtime_db, [
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
    ])

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "anxious",
        "timestamp": "2026-05-19T18:00:00+08:00",
        "turn_count": 3,
    }


def test_load_latest_successful_emotion_skips_malformed_trailing_records(tmp_path, monkeypatch):
    runtime_db = _set_runtime_db(tmp_path, monkeypatch)
    _replace_analysis_records(runtime_db, [
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
        {
            "timestamp": ["2026-05-19T18:30:00+08:00"],
            "turn_count": "6",
            "emotion": "sad",
            "success": True,
        },
        {
            "timestamp": "2026-05-19T18:40:00+08:00",
            "turn_count": True,
            "emotion": "sad",
            "success": True,
        },
    ])

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "anxious",
        "timestamp": "2026-05-19T18:00:00+08:00",
        "turn_count": 3,
    }


def test_load_latest_successful_emotion_returns_none_without_success(tmp_path, monkeypatch):
    runtime_db = _set_runtime_db(tmp_path, monkeypatch)
    _replace_analysis_records(runtime_db, [
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "",
            "success": False,
        }
    ])

    assert emotion.load_latest_successful_emotion() is None


def test_load_latest_successful_emotion_returns_none_for_missing_file(tmp_path, monkeypatch):
    _set_runtime_db(tmp_path, monkeypatch)

    assert emotion.load_latest_successful_emotion() is None


def test_analyze_emotion_persists_structured_state(tmp_path, monkeypatch):
    _set_runtime_db(tmp_path, monkeypatch)

    class FakeLlm:
        def invoke(self, prompt):
            return type("Response", (), {"content": (
                '{"primary_emotion":"anxious","confidence":0.8,'
                '"secondary_emotions":["apprehensive"],'
                '"evidence":"The user is worried.",'
                '"reply_strategy":"Be calm.",'
                '"trajectory_note":"","safety_level":"normal"}'
            )})()

    result = emotion.analyze_emotion(
        FakeLlm(),
        [],
        "I am worried about tomorrow.",
        turn_count=2,
        emotion_interval=2,
    )

    assert result.success is True
    assert result.emotion == "anxious"
    assert result.state.primary_emotion == "anxious"
    data = load_analysis_records()
    assert data[0]["emotion"] == "anxious"
    assert data[0]["dialogue_context"] == "I am worried about tomorrow."
    assert data[0]["state"]["confidence"] == 0.8


def test_emotion_labels_are_shared_from_label_module():
    from chatbot.emotion.labels import EMOTION_LABELS as SHARED_LABELS
    from chatbot.emotion.labels import EMOTION_LABEL_SET as SHARED_LABEL_SET

    assert emotion.EMOTION_LABELS is SHARED_LABELS
    assert emotion.EMOTION_LABEL_SET is SHARED_LABEL_SET
    assert "anxious" in SHARED_LABEL_SET
