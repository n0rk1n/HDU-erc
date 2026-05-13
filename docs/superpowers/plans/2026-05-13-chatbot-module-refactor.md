# Chatbot Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the current single-file chatbot implementation into a small `chatbot` package with separate configuration, LLM, and CLI modules.

**Architecture:** Delete the root `chat.py` entry point and split its existing functions by responsibility. Keep behavior unchanged, use `python -m chatbot.main` as the only supported runtime entry point, and keep tests focused on configuration behavior.

**Tech Stack:** Python 3.13, LangChain, `langchain-openai`, `python-dotenv`, `pytest`.

---

## File Structure

Create these files:

1. `chatbot/__init__.py`：empty package marker.

2. `chatbot/config.py`：configuration dataclass, argument parsing, `.env` loading, and validation.

3. `chatbot/llm.py`：`ChatOpenAI` construction and one-shot answer helper.

4. `chatbot/main.py`：terminal loop and module entry point.

Modify this file:

1. `tests/test_config.py`：import `ConfigError` and `load_config` from `chatbot.config`.

Delete this file:

1. `chat.py`：the old single-file implementation.

## Task 1: Split Chatbot Modules

**Files:**

- Create: `chatbot/__init__.py`

- Create: `chatbot/config.py`

- Create: `chatbot/llm.py`

- Create: `chatbot/main.py`

- Modify: `tests/test_config.py`

- Delete: `chat.py`

- Test: `tests/test_config.py`

- [ ] **Step 1: Create package marker**

Create `chatbot/__init__.py` as an empty file.

- [ ] **Step 2: Move configuration code**

Create `chatbot/config.py` with:

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

- [ ] **Step 3: Move LLM code**

Create `chatbot/llm.py` with:

```python
from langchain_openai import ChatOpenAI

from chatbot.config import ChatConfig


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

- [ ] **Step 4: Move CLI code**

Create `chatbot/main.py` with:

```python
from langchain_openai import ChatOpenAI

from chatbot.config import ConfigError, load_config
from chatbot.llm import ask_once, build_llm


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

- [ ] **Step 5: Update tests**

Change the import in `tests/test_config.py` to:

```python
import pytest

from chatbot.config import ConfigError, load_config
```

Leave the four existing tests unchanged.

- [ ] **Step 6: Delete old entry point**

Delete `chat.py`.

- [ ] **Step 7: Run configuration tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 8: Run syntax check**

Run:

```bash
.venv/bin/python -m py_compile chatbot/__init__.py chatbot/config.py chatbot/llm.py chatbot/main.py tests/test_config.py
```

Expected: command exits successfully.

- [ ] **Step 9: Run missing-key smoke check**

Run:

```bash
OPENAI_API_KEY= .venv/bin/python -m chatbot.main
```

Expected: command exits with status `1` and prints:

```text
Configuration error: OPENAI_API_KEY is required. Set it in .env or your environment.
```

- [ ] **Step 10: Verify old entry point is gone**

Run:

```bash
test ! -f chat.py
```

Expected: command exits successfully.

- [ ] **Step 11: Commit module split**

Run:

```bash
git add chatbot tests/test_config.py chat.py
git commit -m "refactor: split chatbot modules"
```

Expected: commit succeeds on the `dev` branch.

## Task 2: Final Verification

**Files:**

- Verify: `chatbot/__init__.py`

- Verify: `chatbot/config.py`

- Verify: `chatbot/llm.py`

- Verify: `chatbot/main.py`

- Verify: `tests/test_config.py`

- [ ] **Step 1: Run all tests**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all 4 tests pass.

- [ ] **Step 2: Run syntax check**

Run:

```bash
.venv/bin/python -m py_compile chatbot/__init__.py chatbot/config.py chatbot/llm.py chatbot/main.py tests/test_config.py
```

Expected: command exits successfully.

- [ ] **Step 3: Run missing-key smoke check**

Run:

```bash
OPENAI_API_KEY= .venv/bin/python -m chatbot.main
```

Expected: command exits with status `1` and prints:

```text
Configuration error: OPENAI_API_KEY is required. Set it in .env or your environment.
```

- [ ] **Step 4: Verify file layout**

Run:

```bash
test ! -f chat.py
test -f chatbot/__init__.py
test -f chatbot/config.py
test -f chatbot/llm.py
test -f chatbot/main.py
```

Expected: all commands exit successfully.

- [ ] **Step 5: Inspect Git state**

Run:

```bash
git status --short
git branch --show-current
```

Expected: status is clean and current branch is `dev`.

- [ ] **Step 6: Manual real API check**

After creating `.env` from `.env.example` and filling a valid key, run:

```bash
.venv/bin/python -m chatbot.main
```

Ask one short question, then type `quit`.

Expected: the program returns one model answer and exits after `quit`. If no valid API key is configured, skip this step and report that real API verification was not run.
