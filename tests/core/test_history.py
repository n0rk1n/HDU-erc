from pathlib import Path

import pytest

import chatbot.core.history as history
from chatbot.core.runtime_store import RuntimeStore
from chatbot.core.history import (
    REGENERATION_REASONS,
    append_ai_message,
    append_message,
    format_recent,
    load_history,
    prepare_message_regeneration,
    record_message_feedback,
    record_message_regeneration,
)


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    test_db = tmp_path / "runtime.sqlite3"
    monkeypatch.setattr("chatbot.core.history.RUNTIME_DB_PATH", str(test_db))
    return test_db


def _replace_history(db_path, records):
    RuntimeStore(str(db_path)).replace_json_records(history.HISTORY_NAMESPACE, records)


def test_load_history_file_not_found(history_file):
    assert load_history() == []


def test_default_history_file_is_project_data_file():
    path = Path(history.RUNTIME_DB_PATH)

    assert path.is_absolute()
    assert path.name == "runtime.sqlite3"
    assert path.parent.name == "records"
    assert path.parent.parent.name == "data"
    assert path.parent.parent.parent == Path(__file__).resolve().parents[2]


def test_load_history_ignores_legacy_data_file_when_database_missing(tmp_path, monkeypatch):
    runtime_db = tmp_path / "data" / "records" / "runtime.sqlite3"
    legacy_file = tmp_path / "data" / "chat_history.json"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text('[{"role": "human", "content": "hello"}]')
    monkeypatch.setattr("chatbot.core.history.RUNTIME_DB_PATH", str(runtime_db))

    assert load_history() == []


def test_load_history_empty_file(history_file):
    assert load_history() == []


def test_load_history_corrupted_file(history_file):
    history_file.write_text("not sqlite")
    assert load_history() == []


def test_append_message_creates_file(history_file):
    append_message("human", "hello")
    data = load_history()
    assert len(data) == 1
    assert data[0]["role"] == "human"
    assert data[0]["content"] == "hello"
    assert "timestamp" in data[0]


def test_append_message_appends(history_file):
    append_message("human", "msg1")
    append_message("ai", "reply1")
    data = load_history()
    assert len(data) == 2
    assert data[1]["role"] == "ai"
    assert data[1]["content"] == "reply1"


def test_append_ai_message_creates_feedback_ready_record(history_file):
    record = append_ai_message("reply")

    data = load_history()

    assert record == data[0]
    assert data[0]["role"] == "ai"
    assert data[0]["content"] == "reply"
    assert data[0]["id"].startswith("ai_")
    assert data[0]["feedback"] is None
    assert "timestamp" in data[0]


def test_append_message_keeps_human_record_shape(history_file):
    append_message("human", "hello")

    data = load_history()

    assert data[0]["role"] == "human"
    assert data[0]["content"] == "hello"
    assert "timestamp" in data[0]
    assert "id" not in data[0]
    assert "feedback" not in data[0]


def test_load_history_returns_all_records(history_file):
    append_message("human", "q1")
    append_message("ai", "a1")
    records = load_history()
    assert len(records) == 2
    assert records[0]["content"] == "q1"


def test_record_message_feedback_writes_first_rating(history_file):
    record = append_ai_message("reply")

    result = record_message_feedback(record["id"], "like")
    data = load_history()

    assert result.status == "updated"
    assert result.feedback == "like"
    assert data[0]["feedback"] == "like"


def test_record_message_feedback_rejects_second_rating(history_file):
    record = append_ai_message("reply")
    record_message_feedback(record["id"], "like")

    result = record_message_feedback(record["id"], "dislike")
    data = load_history()

    assert result.status == "already_rated"
    assert result.feedback == "like"
    assert data[0]["feedback"] == "like"


def test_record_message_feedback_returns_not_found(history_file):
    result = record_message_feedback("ai_missing", "like")

    assert result.status == "not_found"
    assert result.feedback == ""


def test_record_message_feedback_rejects_human_message(history_file):
    append_message("human", "hello")
    data = load_history()
    data[0]["id"] = "human_1"
    _replace_history(history_file, data)

    result = record_message_feedback("human_1", "like")

    assert result.status == "not_ai"
    assert result.feedback == ""


def test_record_message_feedback_rejects_invalid_value(history_file):
    record = append_ai_message("reply")

    result = record_message_feedback(record["id"], "neutral")

    assert result.status == "invalid_feedback"
    assert result.feedback == ""


def test_prepare_message_regeneration_rejects_invalid_reason(history_file):
    append_message("human", "q1")
    record = append_ai_message("bad answer")

    result = prepare_message_regeneration(record["id"], "too long")

    assert result.status == "invalid_reason"
    assert result.original_user_message == ""


def test_prepare_message_regeneration_returns_not_found(history_file):
    result = prepare_message_regeneration("ai_missing", "不准确")

    assert result.status == "not_found"
    assert result.message_id == ""


def test_prepare_message_regeneration_rejects_human_message(history_file):
    append_message("human", "hello")
    data = load_history()
    data[0]["id"] = "human_1"
    _replace_history(history_file, data)

    result = prepare_message_regeneration("human_1", "不准确")

    assert result.status == "not_ai"
    assert result.message_id == ""


def test_prepare_message_regeneration_rejects_existing_regeneration(history_file):
    append_message("human", "q1")
    original = append_ai_message("bad answer")
    regenerated = record_message_regeneration(
        original["id"],
        "不准确",
        "better answer",
    )

    result = prepare_message_regeneration(original["id"], "不完整")

    assert result.status == "already_regenerated"
    assert result.original_message_id == original["id"]
    assert result.message_id == regenerated.message_id
    assert result.reason == "不准确"
    assert result.original_user_message == "q1"


def test_prepare_message_regeneration_requires_prompt(history_file):
    record = append_ai_message("bad answer")

    result = prepare_message_regeneration(record["id"], "不准确")

    assert result.status == "missing_prompt"
    assert result.original_message_id == record["id"]


def test_prepare_message_regeneration_returns_ready_with_original_prompt(history_file):
    append_message("human", "q1")
    record = append_ai_message("bad answer")

    result = prepare_message_regeneration(record["id"], "不准确")

    assert result.status == "ready"
    assert result.original_message_id == record["id"]
    assert result.reason == "不准确"
    assert result.original_user_message == "q1"


def test_prepare_message_regeneration_uses_original_prompt_for_regenerated_reply(history_file):
    append_message("human", "q1")
    original = append_ai_message("bad answer")
    append_message("human", "q2")
    append_ai_message("later answer")
    regenerated = record_message_regeneration(
        original["id"],
        "不准确",
        "better answer",
    )

    result = prepare_message_regeneration(regenerated.message_id, "不完整")

    assert result.status == "ready"
    assert result.original_message_id == regenerated.message_id
    assert result.original_user_message == "q1"


def test_record_message_regeneration_rejects_invalid_reason(history_file):
    append_message("human", "q1")
    record = append_ai_message("bad answer")

    result = record_message_regeneration(record["id"], "too long", "better answer")

    assert result.status == "invalid_reason"
    assert result.message_id == ""
    assert load_history()[-1]["content"] == "bad answer"


def test_record_message_regeneration_returns_not_found(history_file):
    result = record_message_regeneration("ai_missing", "不准确", "better answer")

    assert result.status == "not_found"
    assert result.message_id == ""


def test_record_message_regeneration_rejects_human_message(history_file):
    append_message("human", "hello")
    data = load_history()
    data[0]["id"] = "human_1"
    _replace_history(history_file, data)

    result = record_message_regeneration("human_1", "不准确", "better answer")

    assert result.status == "not_ai"
    assert result.message_id == ""


def test_record_message_regeneration_requires_preceding_human_prompt(history_file):
    record = append_ai_message("bad answer")

    result = record_message_regeneration(record["id"], "不准确", "better answer")

    assert result.status == "missing_prompt"
    assert result.message_id == ""


def test_record_message_regeneration_records_metadata_and_new_reply(history_file):
    append_message("human", "q1")
    original = append_ai_message("bad answer")

    result = record_message_regeneration(original["id"], "不准确", "better answer")
    data = load_history()

    assert result.status == "updated"
    assert result.reason == "不准确"
    assert result.original_message_id == original["id"]
    assert result.message_id.startswith("ai_")
    assert result.content == "better answer"
    assert result.original_user_message == "q1"
    assert data[1]["regeneration"]["reason"] == "不准确"
    assert data[1]["regeneration"]["regenerated_message_id"] == result.message_id
    assert data[1]["regeneration"]["original_user_message"] == "q1"
    assert data[1]["regeneration"]["original_ai_content"] == "bad answer"
    assert "timestamp" in data[1]["regeneration"]
    assert data[2]["id"] == result.message_id
    assert data[2]["role"] == "ai"
    assert data[2]["content"] == "better answer"
    assert data[2]["feedback"] is None
    assert data[2]["regenerated_from"] == original["id"]


def test_record_message_regeneration_rejects_second_regeneration(history_file):
    append_message("human", "q1")
    original = append_ai_message("bad answer")
    record_message_regeneration(original["id"], "不准确", "better answer")

    result = record_message_regeneration(original["id"], "不完整", "third answer")

    assert result.status == "already_regenerated"
    assert result.message_id
    assert len(load_history()) == 3


def test_record_message_regeneration_uses_original_prompt_for_regenerated_reply(history_file):
    append_message("human", "q1")
    original = append_ai_message("bad answer")
    append_message("human", "q2")
    append_ai_message("later answer")
    regenerated = record_message_regeneration(
        original["id"],
        "不准确",
        "better answer",
    )

    result = record_message_regeneration(
        regenerated.message_id,
        "不完整",
        "third answer",
    )

    assert result.status == "updated"
    assert result.original_message_id == regenerated.message_id
    assert result.original_user_message == "q1"


def test_regeneration_reasons_are_fixed():
    assert REGENERATION_REASONS == {
        "不准确",
        "不完整",
        "没有理解我的问题",
        "语气不合适",
        "其他",
    }


def test_append_message_write_failure_does_not_crash(monkeypatch):
    monkeypatch.setattr(
        "chatbot.core.history.RuntimeStore.append_json_record",
        lambda self, namespace, record: False,
    )

    assert append_message("human", "hello") is None


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
