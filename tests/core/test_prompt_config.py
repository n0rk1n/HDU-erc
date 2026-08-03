import json
from pathlib import Path

from chatbot.core.prompt_config import DEFAULT_PROMPT_CONFIG_PATH, load_prompt_config


def test_default_prompt_config_is_in_project_data_directory():
    path = Path(DEFAULT_PROMPT_CONFIG_PATH)

    assert path.name == "prompts.json"
    assert path.parent.name == "config"
    assert path.parent.parent.name == "data"
    assert path.parent.parent.parent == Path(__file__).resolve().parents[2]


def test_load_prompt_config_returns_defaults_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(tmp_path / "missing.json"))

    config = load_prompt_config()

    assert "gentle emotional companion" in config.chat_system
    assert "Infer the emotion expressed by the target user" in config.emotion_analysis


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
    assert "Infer the emotion expressed by the target user" in config.emotion_analysis
