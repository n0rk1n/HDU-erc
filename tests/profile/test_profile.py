from pathlib import Path

import pytest

import chatbot.profile.repository as profile
from chatbot.core.config import GraphConfig
from chatbot.persistence.runtime import open_persistence
from chatbot.core.runtime_store import RuntimeStore
from chatbot.profile import load_profile, format_profile, save_profile
from chatbot.profile.onboarding import MAX_PROFILE_VALUE_LENGTH

@pytest.fixture
def profile_file(tmp_path, monkeypatch):
    test_db = tmp_path / "runtime.sqlite3"
    monkeypatch.setattr("chatbot.profile.repository.RUNTIME_DB_PATH", str(test_db))
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
    assert path.parent.parent.parent == Path(__file__).resolve().parents[2]


def test_load_profile_ignores_legacy_root_file_when_database_missing(tmp_path, monkeypatch):
    profile_db = tmp_path / "data" / "records" / "runtime.sqlite3"
    legacy_file = tmp_path / "user_profile.json"
    legacy_file.write_text('{"name": "Alice"}')
    monkeypatch.setattr("chatbot.profile.repository.RUNTIME_DB_PATH", str(profile_db))

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


def test_save_profile_sanitizes_direct_call_input(profile_file):
    assert save_profile({
        "preferred_name": "  小明  ",
        "life_stage": "",
        "companion_expectation": 123,
        "response_style": "x" * 500,
        "unknown": "不要保存",
    }) is True

    assert load_profile() == {
        "preferred_name": "小明",
        "response_style": "x" * MAX_PROFILE_VALUE_LENGTH,
    }


@pytest.mark.asyncio
async def test_profile_is_shared_only_inside_client_namespace(tmp_path):
    config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )
    async with open_persistence(config) as handles:
        await save_profile(handles.store, "client-a", {"response_style": "简短"})

        assert await load_profile(handles.store, "client-a") == {"response_style": "简短"}
        assert await load_profile(handles.store, "client-b") == {}


def test_format_profile_empty():
    assert format_profile({}) == ""


def test_format_profile_single_field():
    result = format_profile({"name": "Alice"})
    assert result == "- name: Alice"


def test_format_profile_multiple_fields():
    result = format_profile({"name": "Alice", "age": "28", "mbti": "INTP"})
    assert result == "- name: Alice\n- age: 28\n- mbti: INTP"
