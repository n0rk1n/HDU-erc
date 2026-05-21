import json
from pathlib import Path

import pytest

import chatbot.profile as profile
from chatbot.profile import load_profile, format_profile

PROFILE_FILE = "user_profile.json"


@pytest.fixture
def profile_file(tmp_path, monkeypatch):
    test_file = tmp_path / "user_profile.json"
    monkeypatch.setattr("chatbot.profile.PROFILE_FILE", str(test_file))
    return test_file


def test_load_profile_file_not_found(profile_file):
    assert load_profile() == {}


def test_default_profile_file_is_project_config_file():
    path = Path(profile.PROFILE_FILE)

    assert path.is_absolute()
    assert path.name == "user_profile.json"
    assert path.parent.name == "config"
    assert path.parent.parent.name == "data"
    assert path.parent.parent.parent == Path(__file__).resolve().parents[1]


def test_load_profile_reads_legacy_root_file_when_new_file_missing(tmp_path, monkeypatch):
    profile_file = tmp_path / "data" / "config" / "user_profile.json"
    legacy_file = tmp_path / "user_profile.json"
    legacy_file.write_text(json.dumps({"name": "Alice"}))
    monkeypatch.setattr("chatbot.profile.PROFILE_FILE", str(profile_file))
    monkeypatch.setattr("chatbot.profile.LEGACY_PROFILE_FILE", str(legacy_file))

    assert load_profile() == {"name": "Alice"}


def test_load_profile_empty_file(profile_file):
    profile_file.write_text("")
    assert load_profile() == {}


def test_load_profile_corrupted_file(profile_file):
    profile_file.write_text("not valid json")
    assert load_profile() == {}


def test_load_profile_non_dict_json(profile_file):
    profile_file.write_text("[]")
    assert load_profile() == {}


def test_load_profile_returns_dict(profile_file):
    data = {"name": "Alice", "age": "28", "mbti": "INTP"}
    profile_file.write_text(json.dumps(data))
    assert load_profile() == data


def test_load_profile_skips_empty_values(profile_file):
    data = {"name": "Alice", "age": "", "mbti": "INTP"}
    profile_file.write_text(json.dumps(data))
    result = load_profile()
    assert result == {"name": "Alice", "mbti": "INTP"}


def test_format_profile_empty():
    assert format_profile({}) == ""


def test_format_profile_single_field():
    result = format_profile({"name": "Alice"})
    assert result == "- name: Alice"


def test_format_profile_multiple_fields():
    result = format_profile({"name": "Alice", "age": "28", "mbti": "INTP"})
    assert result == "- name: Alice\n- age: 28\n- mbti: INTP"
