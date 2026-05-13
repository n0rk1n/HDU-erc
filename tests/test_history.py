import json
from pathlib import Path

import pytest

from chatbot.history import HISTORY_FILE, append_message, load_history


def test_load_history_file_not_found(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    assert load_history() == []


def test_load_history_corrupted_file(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    test_file.write_text("not valid json")
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    assert load_history() == []


def test_append_message_creates_file(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    append_message("human", "hello")
    data = json.loads(test_file.read_text())
    assert len(data) == 1
    assert data[0]["role"] == "human"
    assert data[0]["content"] == "hello"
    assert "timestamp" in data[0]


def test_append_message_appends(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    append_message("human", "msg1")
    append_message("ai", "reply1")
    data = json.loads(test_file.read_text())
    assert len(data) == 2
    assert data[1]["role"] == "ai"
    assert data[1]["content"] == "reply1"


def test_load_history_returns_all_records(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    append_message("human", "q1")
    append_message("ai", "a1")
    records = load_history()
    assert len(records) == 2
    assert records[0]["content"] == "q1"
