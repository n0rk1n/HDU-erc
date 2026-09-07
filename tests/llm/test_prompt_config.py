from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot.core.errors import ConfigError
from chatbot.llm.prompt import (
    DEFAULT_PROMPTS_CONFIG_PATH,
    DEFAULT_SYSTEM_PROMPT,
    build_prompt,
    get_system_prompt,
)


def test_repo_default_config_matches_built_in_system_prompt() -> None:
    data = json.loads(DEFAULT_PROMPTS_CONFIG_PATH.read_text(encoding="utf-8"))

    assert data["system"] == DEFAULT_SYSTEM_PROMPT


def test_get_system_prompt_returns_default_when_config_file_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import chatbot.llm.prompt as prompt_module

    monkeypatch.delenv("CHAT_SYSTEM_PROMPT_PATH", raising=False)
    monkeypatch.setattr(
        prompt_module,
        "DEFAULT_PROMPTS_CONFIG_PATH",
        tmp_path / "does-not-exist.json",
    )

    assert get_system_prompt() == DEFAULT_SYSTEM_PROMPT


def test_get_system_prompt_returns_default_when_default_config_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import chatbot.llm.prompt as prompt_module

    empty_config = tmp_path / "chat_prompts.json"
    empty_config.write_text("", encoding="utf-8")
    monkeypatch.delenv("CHAT_SYSTEM_PROMPT_PATH", raising=False)
    monkeypatch.setattr(prompt_module, "DEFAULT_PROMPTS_CONFIG_PATH", empty_config)

    assert get_system_prompt() == DEFAULT_SYSTEM_PROMPT


def test_get_system_prompt_loads_system_from_explicit_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "chat_prompts.json"
    config_path.write_text(
        json.dumps({"version": "v2", "system": "自定义系统提示"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHAT_SYSTEM_PROMPT_PATH", str(config_path))

    assert get_system_prompt() == "自定义系统提示"


def test_build_prompt_prepends_configured_system_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "chat_prompts.json"
    config_path.write_text(
        json.dumps({"version": "v2", "system": "仅回答具体问题"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHAT_SYSTEM_PROMPT_PATH", str(config_path))

    prompt = build_prompt([{"role": "user", "content": "你好"}])

    assert prompt[0].type == "system"
    assert prompt[0].content.startswith("仅回答具体问题\n\n")
    assert '{"messages":["完整气泡正文"]}' in prompt[0].content
    assert prompt[1].content == "你好"


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps(["not", "an", "object"]),
        json.dumps({"other_key": "value"}),
        json.dumps({"version": "v2", "system": ""}),
        json.dumps({"version": "v2", "system": 42}),
    ],
)
def test_invalid_explicit_config_raises_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    config_path = tmp_path / "chat_prompts.json"
    config_path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("CHAT_SYSTEM_PROMPT_PATH", str(config_path))

    with pytest.raises(ConfigError):
        get_system_prompt()


def test_explicit_missing_config_path_raises_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "does-not-exist.json"
    monkeypatch.setenv("CHAT_SYSTEM_PROMPT_PATH", str(missing))

    with pytest.raises(ConfigError):
        get_system_prompt()
