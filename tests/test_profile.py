from pathlib import Path

import pytest

import chatbot.profile as profile
from chatbot.runtime_store import RuntimeStore
from chatbot.profile import load_profile, format_profile, save_profile

@pytest.fixture
def profile_file(tmp_path, monkeypatch):
    test_db = tmp_path / "runtime.sqlite3"
    monkeypatch.setattr("chatbot.profile.RUNTIME_DB_PATH", str(test_db))
    return test_db


def _replace_profile(db_path, data):
    RuntimeStore(str(db_path)).replace_profile(data)


def test_load_profile_file_not_found(profile_file):
    assert load_profile() == {}


def test_default_profile_file_is_project_config_file():
    path = Path(profile.RUNTIME_DB_PATH)

    assert path.is_absolute()
    assert path.name == "runtime.sqlite3"
    assert path.parent.name == "records"
    assert path.parent.parent.name == "data"
    assert path.parent.parent.parent == Path(__file__).resolve().parents[1]


def test_load_profile_ignores_legacy_root_file_when_database_missing(tmp_path, monkeypatch):
    profile_db = tmp_path / "data" / "records" / "runtime.sqlite3"
    legacy_file = tmp_path / "user_profile.json"
    legacy_file.write_text('{"name": "Alice"}')
    monkeypatch.setattr("chatbot.profile.RUNTIME_DB_PATH", str(profile_db))

    assert load_profile() == {}


def test_load_profile_empty_file(profile_file):
    assert load_profile() == {}


def test_load_profile_corrupted_file(profile_file):
    profile_file.write_text("not sqlite")
    assert load_profile() == {}


def test_load_profile_non_dict_json(profile_file):
    assert load_profile() == {}


def test_load_profile_returns_dict(profile_file):
    data = {"name": "Alice", "age": "28", "mbti": "INTP"}
    _replace_profile(profile_file, data)
    assert load_profile() == data


def test_load_profile_skips_empty_values(profile_file):
    data = {"name": "Alice", "age": "", "mbti": "INTP"}
    _replace_profile(profile_file, data)
    result = load_profile()
    assert result == {"name": "Alice", "mbti": "INTP"}


def test_save_profile_writes_database_profile(profile_file):
    assert save_profile({"preferred_name": "小明", "response_style": "简短"}) is True

    assert load_profile() == {
        "preferred_name": "小明",
        "response_style": "简短",
    }


def test_format_profile_empty():
    assert format_profile({}) == ""


def test_format_profile_single_field():
    result = format_profile({"name": "Alice"})
    assert result == "- name: Alice"


def test_format_profile_multiple_fields():
    result = format_profile({"name": "Alice", "age": "28", "mbti": "INTP"})
    assert result == "- name: Alice\n- age: 28\n- mbti: INTP"
