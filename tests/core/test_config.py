from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from chatbot.core.config import AppConfig
from chatbot.core.errors import ConfigError


CONFIG_ENVIRONMENT = {
    "LLM_API_KEY": "test-api-key",
    "LLM_MODEL": "test-model",
    "LLM_BASE_URL": "https://llm.example.test/v1",
    "LLM_TEMPERATURE": "0.25",
    "LLM_TIMEOUT_SECONDS": "12.5",
    "CHAT_CONTEXT_MESSAGE_LIMIT": "32",
    "SQLITE_DB_PATH": "data/test-chatbot.sqlite3",
}


def set_valid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in CONFIG_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def test_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches accepting a startup configuration that cannot call the model."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        AppConfig.from_env()


def test_config_reads_all_values_and_hides_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a wrong environment mapping or exposing the model credential in repr."""
    set_valid_environment(monkeypatch)

    config = AppConfig.from_env()

    assert config.llm_api_key.get_secret_value() == "test-api-key"
    assert config.llm_model == "test-model"
    assert config.llm_base_url == "https://llm.example.test/v1"
    assert config.llm_temperature == 0.25
    assert config.llm_timeout_seconds == 12.5
    assert config.context_message_limit == 32
    assert config.sqlite_db_path == Path("data/test-chatbot.sqlite3")
    assert "test-api-key" not in repr(config)
    with pytest.raises(FrozenInstanceError):
        config.llm_model = "another-model"  # type: ignore[misc]


def test_config_uses_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches missing defaults when only the required API key is configured."""
    for name in CONFIG_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    config = AppConfig.from_env()

    assert config.llm_model == "gpt-4o-mini"
    assert config.llm_base_url is None
    assert config.llm_temperature == 0.7
    assert config.llm_timeout_seconds == 60.0
    assert config.context_message_limit == 40
    assert config.sqlite_db_path == Path("data/chatbot.sqlite3")


@pytest.mark.parametrize("value", ["0", "750", "10000"])
def test_bubble_gap_env_is_loaded(monkeypatch, value):
    set_valid_environment(monkeypatch)
    monkeypatch.setenv("CHAT_BUBBLE_GAP_MS", value)
    assert AppConfig.from_env().chat_bubble_gap_ms == int(value)


@pytest.mark.parametrize("value", ["-1", "10001", "1.5", "nan"])
def test_invalid_bubble_gap_stops_startup(monkeypatch, value):
    set_valid_environment(monkeypatch)
    monkeypatch.setenv("CHAT_BUBBLE_GAP_MS", value)
    with pytest.raises(ConfigError, match="CHAT_BUBBLE_GAP_MS"):
        AppConfig.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_TEMPERATURE", "-0.01"),
        ("LLM_TEMPERATURE", "2.01"),
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("CHAT_CONTEXT_MESSAGE_LIMIT", "0"),
        ("CHAT_CONTEXT_MESSAGE_LIMIT", "201"),
    ],
)
def test_config_rejects_out_of_range_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Catches invalid model settings reaching runtime model and context handling."""
    set_valid_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError, match=name):
        AppConfig.from_env()
