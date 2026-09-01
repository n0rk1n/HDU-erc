import pytest

from chatbot.core.config import GraphConfig
from chatbot.persistence.runtime import open_persistence
from chatbot.profile import load_profile, format_profile, save_profile
from chatbot.profile.onboarding import MAX_PROFILE_VALUE_LENGTH

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
        await save_profile(
            handles.store,
            "client-a",
            {
                "preferred_name": "  小明  ",
                "life_stage": "",
                "companion_expectation": 123,
                "response_style": "x" * 500,
                "unknown": "不要保存",
            },
        )

        assert await load_profile(handles.store, "client-a") == {
            "preferred_name": "小明",
            "response_style": "x" * MAX_PROFILE_VALUE_LENGTH,
        }
        assert await load_profile(handles.store, "client-b") == {}


def test_format_profile_empty():
    assert format_profile({}) == ""


def test_format_profile_single_field():
    result = format_profile({"name": "Alice"})
    assert result == "- name: Alice"


def test_format_profile_multiple_fields():
    result = format_profile({"name": "Alice", "age": "28", "mbti": "INTP"})
    assert result == "- name: Alice\n- age: 28\n- mbti: INTP"
