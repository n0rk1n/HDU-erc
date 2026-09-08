import pytest
from emotion_lab.config import validate_config


def test_credentials_in_run_config_are_rejected():
    with pytest.raises(ValueError):
        validate_config({"method": "zero-shot", "api_key": "secret"})


def test_existing_emotion_environment_overrides_general_defaults(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", ".7")
    monkeypatch.setenv("EMOTION_LLM_TEMPERATURE", ".2")
    monkeypatch.setenv("EMOTION_LLM_TIMEOUT_SECONDS", "90.5")
    monkeypatch.setenv("EMOTION_CONTEXT_TOKENS", "100000")
    monkeypatch.setenv("EMOTION_OUTPUT_TOKENS", "1234")
    value = validate_config({"method": "zero-shot"})
    assert value["model"]["temperature"] == 0.2
    assert value["model"]["timeout_seconds"] == 90.5
    assert value["model"]["context_tokens"] == 100000
    assert value["model"]["max_tokens"] == 1234
