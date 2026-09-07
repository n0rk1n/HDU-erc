import json

import pytest

from chatbot.core.errors import ConfigError
from chatbot.emotion.prompt import get_emotion_prompt
from chatbot.llm.prompt import get_system_prompt


@pytest.fixture(params=["chat", "emotion"])
def read_prompt(request, tmp_path, monkeypatch):
    path = tmp_path / "prompt.json"
    if request.param == "chat":
        monkeypatch.setenv("CHAT_SYSTEM_PROMPT_PATH", str(path))
        load = get_system_prompt
    else:
        monkeypatch.setenv("EMOTION_SYSTEM_PROMPT_PATH", str(path))
        load = lambda: get_emotion_prompt()["system"]
    return path, load


def test_both_consumers_reload_unified_config_without_changing_text(read_prompt):
    path, load = read_prompt
    for text in ("  第一行\n第二行\n", "修改后的提示词"):
        path.write_text(json.dumps({"version": "custom-1", "system": text}))
        assert load() == text


@pytest.mark.parametrize("content", [
    "", "{", "[]",
    '{"system":"rules"}',
    '{"version":"v"}',
    '{"version":null,"system":"rules"}',
    '{"version":2,"system":"rules"}',
    '{"version":" ","system":"rules"}',
    '{"version":"v","system":null}',
    '{"version":"v","system":[]}',
    '{"version":"v","system":" "}',
    '{"version":"v","version":"other","system":"rules"}',
    '{"version":"v","system":"first","system":"second"}',
    '{"version":"v","chat_system":"rules"}',
    '{"version":"v","emotion_system":"rules"}',
])
def test_both_consumers_reject_invalid_or_legacy_config(read_prompt, content):
    path, load = read_prompt
    path.write_text(content)
    with pytest.raises(ConfigError):
        load()


def test_both_consumers_reject_unreadable_encoding(read_prompt):
    path, load = read_prompt
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigError):
        load()
