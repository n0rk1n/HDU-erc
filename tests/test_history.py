import json
from pathlib import Path

import pytest

import chatbot.history as history
from chatbot.history import append_message, format_recent, load_history


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    return test_file


def test_load_history_file_not_found(history_file):
    assert load_history() == []


def test_default_history_file_is_project_data_file():
    path = Path(history.HISTORY_FILE)

    assert path.is_absolute()
    assert path.name == "chat_history.json"
    assert path.parent.name == "data"
    assert path.parent.parent == Path(__file__).resolve().parents[1]


def test_load_history_empty_file(history_file):
    history_file.write_text("")
    assert load_history() == []


def test_load_history_corrupted_file(history_file):
    history_file.write_text("not valid json")
    assert load_history() == []


def test_append_message_creates_file(history_file):
    append_message("human", "hello")
    data = json.loads(history_file.read_text())
    assert len(data) == 1
    assert data[0]["role"] == "human"
    assert data[0]["content"] == "hello"
    assert "timestamp" in data[0]


def test_append_message_appends(history_file):
    append_message("human", "msg1")
    append_message("ai", "reply1")
    data = json.loads(history_file.read_text())
    assert len(data) == 2
    assert data[1]["role"] == "ai"
    assert data[1]["content"] == "reply1"


def test_load_history_returns_all_records(history_file):
    append_message("human", "q1")
    append_message("ai", "a1")
    records = load_history()
    assert len(records) == 2
    assert records[0]["content"] == "q1"


def test_append_message_write_failure_does_not_crash(history_file, capsys):
    """Write failure prints a warning and does not raise."""
    # Make directory read-only to force write failure
    Path(history_file).parent.chmod(0o444)
    try:
        append_message("human", "hello")  # should not raise
    finally:
        Path(history_file).parent.chmod(0o755)


def test_format_recent_empty():
    assert format_recent([]) == ""


def test_format_recent_human_message():
    records = [{"role": "human", "content": "hello"}]
    assert format_recent(records) == "You: hello"


def test_format_recent_ai_message():
    records = [{"role": "ai", "content": "hi there"}]
    assert format_recent(records) == "Bot: hi there"


def test_format_recent_multiple_messages():
    records = [
        {"role": "human", "content": "q1"},
        {"role": "ai", "content": "a1"},
        {"role": "human", "content": "q2"},
        {"role": "ai", "content": "a2"},
    ]
    expected = "You: q1\nBot: a1\nYou: q2\nBot: a2"
    assert format_recent(records) == expected


def test_format_recent_returns_last_n():
    records = [
        {"role": "human", "content": "q1"},
        {"role": "ai", "content": "a1"},
        {"role": "human", "content": "q2"},
        {"role": "ai", "content": "a2"},
    ]
    expected = "You: q2\nBot: a2"
    assert format_recent(records, n=2) == expected


def test_format_recent_n_exceeds_length():
    records = [
        {"role": "human", "content": "q1"},
        {"role": "ai", "content": "a1"},
    ]
    expected = "You: q1\nBot: a1"
    assert format_recent(records, n=10) == expected
