# LangChain CLI Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal LangChain CLI chatbot that reads OpenAI configuration from `.env`, supports command-line overrides, and performs independent one-question, one-answer model calls.

**Architecture:** Keep the implementation in one small `chat.py` module because this is a minimal CLI tool. The module exposes focused functions for configuration loading, model construction, one-off question answering, and terminal interaction so tests can cover configuration behavior without calling the OpenAI API.

**Tech Stack:** Python 3.13, LangChain, `langchain-openai`, `python-dotenv`, `pytest`.

---

## File Structure

Create or modify these files:

1. `requirements.txt`：运行和测试依赖。

2. `.env.example`：OpenAI 配置示例。

3. `chat.py`：CLI 入口、配置加载、LangChain 模型构建、单次问答和终端循环。

4. `tests/test_config.py`：配置加载和校验测试，不真实调用 OpenAI API。

## Task 1: Dependencies And Environment Example

**Files:**

- Create: `requirements.txt`

- Create: `.env.example`

- Test: manual file inspection

- [ ] **Step 1: Create dependency file**

Write `requirements.txt` with exactly:

```text
langchain
langchain-openai
python-dotenv
pytest
```

- [ ] **Step 2: Create environment example**

Write `.env.example` with exactly:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
OPENAI_TEMPERATURE=0.7
```

- [ ] **Step 3: Install dependencies**

Run:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Expected: dependencies install successfully into the project virtual environment.

- [ ] **Step 4: Verify files**

Run:

```bash
sed -n '1,80p' requirements.txt
sed -n '1,80p' .env.example
```

Expected: both files print the exact content shown above.

- [ ] **Step 5: Commit**

Run:

```bash
git add requirements.txt .env.example
git commit -m "chore: add chatbot dependencies"
```

Expected: commit succeeds on the `dev` branch.

## Task 2: Configuration Tests

**Files:**

- Create: `tests/test_config.py`

- Create: `chat.py`

- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_config.py` with:

```python
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
```

- [ ] **Step 2: Add minimal import targets**

Create `chat.py` with:

```python
class ConfigError(ValueError):
    pass


def load_config(argv=None, *, load_env=True):
    raise NotImplementedError("Configuration loading is not implemented yet.")
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: tests fail because `load_config()` raises `NotImplementedError`.

- [ ] **Step 4: Commit failing tests**

Run:

```bash
git add tests/test_config.py chat.py
git commit -m "test: cover chatbot configuration"
```

Expected: commit succeeds with failing tests intentionally documented in history.

## Task 3: Configuration Implementation

**Files:**

- Modify: `chat.py`

- Test: `tests/test_config.py`

- [ ] **Step 1: Implement configuration loading**

Replace `chat.py` with:

```python
import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ChatConfig:
    api_key: str
    model: str
    temperature: float
    base_url: str | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chat with an OpenAI model through LangChain.")
    parser.add_argument("--model", help="OpenAI model name. Overrides OPENAI_MODEL.")
    parser.add_argument("--temperature", help="Sampling temperature. Overrides OPENAI_TEMPERATURE.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Overrides OPENAI_BASE_URL.")
    return parser.parse_args(argv)


def parse_temperature(raw_value: str) -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError("OPENAI_TEMPERATURE must be a number, such as 0.7.") from exc


def load_config(argv=None, *, load_env=True) -> ChatConfig:
    if load_env:
        load_dotenv()
    args = parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("OPENAI_API_KEY is required. Set it in .env or your environment.")

    model = args.model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    raw_temperature = args.temperature or os.getenv("OPENAI_TEMPERATURE") or str(DEFAULT_TEMPERATURE)
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or None
    if base_url is not None:
        base_url = base_url.strip() or None

    return ChatConfig(
        api_key=api_key,
        model=model,
        temperature=parse_temperature(raw_temperature),
        base_url=base_url,
    )
```

- [ ] **Step 2: Run configuration tests**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 3: Commit configuration implementation**

Run:

```bash
git add chat.py tests/test_config.py
git commit -m "feat: load chatbot configuration"
```

Expected: commit succeeds on the `dev` branch.

## Task 4: LangChain One-Shot Chat

**Files:**

- Modify: `chat.py`

- Test: `tests/test_config.py`

- [ ] **Step 1: Add model construction and one-shot answering**

Update `chat.py` so the imports and new functions include:

```python
import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
```

Add these functions after `load_config()`:

```python
def build_llm(config: ChatConfig) -> ChatOpenAI:
    kwargs = {
        "api_key": config.api_key,
        "model": config.model,
        "temperature": config.temperature,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)


def ask_once(llm: ChatOpenAI, question: str) -> str:
    response = llm.invoke(question)
    content = response.content
    if isinstance(content, str):
        return content
    return str(content)
```

- [ ] **Step 2: Run existing tests**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: all 4 tests still pass.

- [ ] **Step 3: Verify imports**

Run:

```bash
python -m py_compile chat.py
```

Expected: command exits successfully.

- [ ] **Step 4: Commit model functions**

Run:

```bash
git add chat.py
git commit -m "feat: add one-shot langchain chat"
```

Expected: commit succeeds on the `dev` branch.

## Task 5: CLI Loop And Error Handling

**Files:**

- Modify: `chat.py`

- Test: manual CLI smoke checks

- [ ] **Step 1: Add terminal loop**

Append these functions to `chat.py`:

```python
def run_chat_loop(llm: ChatOpenAI) -> None:
    print("LangChain CLI chatbot")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        try:
            answer = ask_once(llm, question)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Bot: {answer}")


def main(argv=None) -> int:
    try:
        config = load_config(argv)
        llm = build_llm(config)
        run_chat_loop(llm)
        return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nBye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 3: Run missing-key smoke check**

Run:

```bash
OPENAI_API_KEY= python chat.py
```

Expected: prints a configuration error mentioning `OPENAI_API_KEY` and exits with status `1`.

- [ ] **Step 4: Run syntax check**

Run:

```bash
python -m py_compile chat.py
```

Expected: command exits successfully.

- [ ] **Step 5: Commit CLI loop**

Run:

```bash
git add chat.py
git commit -m "feat: add chatbot cli loop"
```

Expected: commit succeeds on the `dev` branch.

## Task 6: Final Verification

**Files:**

- Verify: `chat.py`

- Verify: `requirements.txt`

- Verify: `.env.example`

- Verify: `tests/test_config.py`

- [ ] **Step 1: Run all tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile chat.py tests/test_config.py
```

Expected: command exits successfully.

- [ ] **Step 3: Inspect Git state**

Run:

```bash
git status --short
git branch --show-current
```

Expected: status is clean and current branch is `dev`.

- [ ] **Step 4: Manual real API check**

After installing dependencies and creating `.env` from `.env.example`, run:

```bash
python chat.py
```

Ask one short question, then type `quit`.

Expected: the program returns one model answer, does not refer to previous questions, and exits after `quit`.
