import json

from chatbot.prompt_config import load_prompt_config


def test_load_prompt_config_returns_defaults_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(tmp_path / "missing.json"))

    config = load_prompt_config()

    assert "gentle emotional companion" in config.chat_system
    assert "Infer the user's current emotion" in config.emotion_analysis


def test_load_prompt_config_overrides_non_empty_prompt_values(tmp_path, monkeypatch):
    config_file = tmp_path / "prompts.json"
    config_file.write_text(
        json.dumps({
            "chat_system": "Custom chat system.",
            "emotion_analysis": "Custom emotion prompt: {dialogue_context}",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(config_file))

    config = load_prompt_config()

    assert config.chat_system == "Custom chat system."
    assert config.emotion_analysis == "Custom emotion prompt: {dialogue_context}"


def test_load_prompt_config_keeps_defaults_for_empty_or_invalid_values(tmp_path, monkeypatch):
    config_file = tmp_path / "prompts.json"
    config_file.write_text(
        json.dumps({
            "chat_system": "",
            "emotion_analysis": ["not", "a", "string"],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(config_file))

    config = load_prompt_config()

    assert "gentle emotional companion" in config.chat_system
    assert "Infer the user's current emotion" in config.emotion_analysis
