import pytest

from chatbot.config import ConfigError, load_config


def test_load_config_uses_chat_llm_environment_values(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "chat-key")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.25")
    monkeypatch.setenv("EMOTION_INTERVAL", "3")
    monkeypatch.delenv("EMOTION_LLM_MODEL", raising=False)

    config = load_config([], load_env=False)

    assert config.chat_llm.api_key == "chat-key"
    assert config.chat_llm.provider == "deepseek"
    assert config.chat_llm.model == "deepseek-chat"
    assert config.chat_llm.base_url == "https://api.deepseek.com/v1"
    assert config.chat_llm.temperature == 0.25
    assert config.emotion_llm == config.chat_llm
    assert config.emotion_interval == 3


def test_legacy_openai_environment_values_still_work(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example.com/v1")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.1")

    config = load_config([], load_env=False)

    assert config.chat_llm.provider == "openai"
    assert config.chat_llm.api_key == "legacy-key"
    assert config.chat_llm.model == "gpt-test"
    assert config.chat_llm.base_url == "https://legacy.example.com/v1"
    assert config.chat_llm.temperature == 0.1


def test_emotion_llm_inherits_chat_values_and_overrides_configured_values(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "chat-key")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("EMOTION_LLM_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("EMOTION_LLM_TEMPERATURE", "0")

    config = load_config([], load_env=False)

    assert config.chat_llm.model == "deepseek-chat"
    assert config.emotion_llm.provider == "deepseek"
    assert config.emotion_llm.api_key == "chat-key"
    assert config.emotion_llm.model == "deepseek-reasoner"
    assert config.emotion_llm.base_url == "https://api.deepseek.com/v1"
    assert config.emotion_llm.temperature == 0.0
    assert config.emotion_llm != config.chat_llm


def test_command_line_values_override_chat_and_emotion_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "chat-key")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "env-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.example.com/v1")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("EMOTION_LLM_MODEL", "env-emotion")
    monkeypatch.setenv("EMOTION_INTERVAL", "9")

    config = load_config(
        [
            "--provider",
            "openai",
            "--model",
            "cli-chat",
            "--temperature",
            "0.9",
            "--base-url",
            "https://cli.example.com/v1",
            "--emotion-provider",
            "deepseek",
            "--emotion-model",
            "cli-emotion",
            "--emotion-temperature",
            "0.2",
            "--emotion-base-url",
            "https://emotion.example.com/v1",
            "--emotion-interval",
            "4",
        ],
        load_env=False,
    )

    assert config.chat_llm.provider == "openai"
    assert config.chat_llm.model == "cli-chat"
    assert config.chat_llm.base_url == "https://cli.example.com/v1"
    assert config.chat_llm.temperature == 0.9
    assert config.emotion_llm.provider == "deepseek"
    assert config.emotion_llm.model == "cli-emotion"
    assert config.emotion_llm.base_url == "https://emotion.example.com/v1"
    assert config.emotion_llm.temperature == 0.2
    assert config.emotion_interval == 4


def test_default_emotion_interval(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMOTION_INTERVAL", raising=False)

    config = load_config([], load_env=False)

    assert config.emotion_interval == 5


def test_missing_chat_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load_config([], load_env=False)


def test_invalid_temperature_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "warm")

    with pytest.raises(ConfigError, match="LLM_TEMPERATURE"):
        load_config([], load_env=False)


def test_invalid_emotion_temperature_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMOTION_LLM_MODEL", "emotion-model")
    monkeypatch.setenv("EMOTION_LLM_TEMPERATURE", "cold")

    with pytest.raises(ConfigError, match="EMOTION_LLM_TEMPERATURE"):
        load_config([], load_env=False)


def test_invalid_emotion_interval_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMOTION_INTERVAL", "soon")

    with pytest.raises(ConfigError, match="EMOTION_INTERVAL"):
        load_config([], load_env=False)


def test_non_positive_emotion_interval_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMOTION_INTERVAL", "0")

    with pytest.raises(ConfigError, match="EMOTION_INTERVAL"):
        load_config([], load_env=False)
