import pytest


@pytest.fixture(autouse=True)
def isolate_local_prompt_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(tmp_path / "missing-prompts.json"))
