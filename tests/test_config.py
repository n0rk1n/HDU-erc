import pytest

from chat import ConfigError, load_config


def test_load_config_uses_environment_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.25")

    config = load_config([], load_env=False)

    assert config.api_key == "test-key"
    assert config.model == "gpt-test"
    assert config.base_url == "https://example.com/v1"
    assert config.temperature == 0.25


def test_command_line_values_override_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.1")

    config = load_config(
        [
            "--model",
            "cli-model",
            "--temperature",
            "0.9",
            "--base-url",
            "https://cli.example.com/v1",
        ],
        load_env=False,
    )

    assert config.api_key == "test-key"
    assert config.model == "cli-model"
    assert config.base_url == "https://cli.example.com/v1"
    assert config.temperature == 0.9


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_config([], load_env=False)


def test_invalid_temperature_raises_clear_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "warm")

    with pytest.raises(ConfigError, match="OPENAI_TEMPERATURE"):
        load_config([], load_env=False)
