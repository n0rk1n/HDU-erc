# JSON 聊天记录持久化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `chatbot/history.py` 模块，实现聊天记录以 JSON 文件持久化，启动时加载到记忆中。

**Architecture:** `history.py` 负责 JSON 文件的读写（每条记录：role、content、timestamp）。`llm.py` 新增 `init_session_history()` 把记录灌入 `InMemoryChatMessageHistory`。`main.py` 在启动时加载历史、每轮对话后追加记录。

**Tech Stack:** Python 3.13 标准库（json, datetime, pathlib）

---

### Task 1: 创建 `chatbot/history.py` — JSON 文件读写

**Files:**
- Create: `chatbot/history.py`
- Test: `tests/test_history.py`

- [ ] **Step 1: 编写测试 `tests/test_history.py`**

```python
import json
from pathlib import Path

import pytest

from chatbot.history import HISTORY_FILE, append_message, load_history


def test_load_history_file_not_found(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    assert load_history() == []


def test_load_history_corrupted_file(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    test_file.write_text("not valid json")
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    assert load_history() == []


def test_append_message_creates_file(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    append_message("human", "hello")
    data = json.loads(test_file.read_text())
    assert len(data) == 1
    assert data[0]["role"] == "human"
    assert data[0]["content"] == "hello"
    assert "timestamp" in data[0]


def test_append_message_appends(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    append_message("human", "msg1")
    append_message("ai", "reply1")
    data = json.loads(test_file.read_text())
    assert len(data) == 2
    assert data[1]["role"] == "ai"
    assert data[1]["content"] == "reply1"


def test_load_history_returns_all_records(tmp_path: Path):
    monkeypatch = pytest.MonkeyPatch()
    test_file = tmp_path / "chat_history.json"
    monkeypatch.setattr("chatbot.history.HISTORY_FILE", str(test_file))
    append_message("human", "q1")
    append_message("ai", "a1")
    records = load_history()
    assert len(records) == 2
    assert records[0]["content"] == "q1"
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `.venv/bin/python -m pytest tests/test_history.py -v`
Expected: ModuleNotFoundError / ImportError (module doesn't exist yet)

- [ ] **Step 3: 实现 `chatbot/history.py`**

```python
import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = "chat_history.json"


def load_history() -> list[dict]:
    path = Path(HISTORY_FILE)
    if not path.exists():
        return []
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_message(role: str, content: str) -> None:
    try:
        records = load_history()
        records.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        path = Path(HISTORY_FILE)
        with path.open("w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"Warning: could not save chat history: {exc}")
```

- [ ] **Step 4: 运行测试，确认全部通过**

Run: `.venv/bin/python -m pytest tests/test_history.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: 验证不影响现有测试**

Run: `.venv/bin/python -m pytest -v`
Expected: 9 passed (4 existing + 5 new)

- [ ] **Step 6: Commit**

```bash
git add chatbot/history.py tests/test_history.py
git commit -m "feat: add JSON chat history persistence module"
```

### Task 2: 新增 `init_session_history()` 到 `chatbot/llm.py`

**Files:**
- Modify: `chatbot/llm.py`

- [ ] **Step 1: 新增 `init_session_history()` 函数**

在 `build_chain()` 函数之前添加：

```python
def init_session_history(session_id: str, records: list[dict]) -> None:
    history = get_session_history(session_id)
    for record in records:
        role = record.get("role")
        content = record.get("content", "")
        if role == "human":
            history.add_user_message(content)
        elif role == "ai":
            history.add_ai_message(content)
```

- [ ] **Step 2: 验证语法正确**

Run: `.venv/bin/python -m py_compile chatbot/llm.py`
Expected: exit code 0, no output

- [ ] **Step 3: Commit**

```bash
git add chatbot/llm.py
git commit -m "feat: add init_session_history function"
```

### Task 3: 修改 `chatbot/main.py` — 加载历史并在对话中追加记录

**Files:**
- Modify: `chatbot/main.py`

- [ ] **Step 1: 添加 import 并更新 `main()` 和 `run_chat_loop()`**

修改 import 行。把：
```python
from chatbot.llm import build_chain, build_llm
```
改为：
```python
from chatbot.history import append_message, load_history
from chatbot.llm import build_chain, build_llm, init_session_history
```

修改 `main()` 以在构建 chain 之前加载历史。把：
```python
def main(argv=None) -> int:
    try:
        config = load_config(argv)
        llm = build_llm(config)
        chain = build_chain(llm)
        run_chat_loop(chain)
        return 0
```
改为：
```python
def main(argv=None) -> int:
    try:
        config = load_config(argv)
        records = load_history()
        llm = build_llm(config)
        init_session_history("default", records)
        chain = build_chain(llm)
        run_chat_loop(chain)
        return 0
```

修改 `run_chat_loop()` 以在每轮追加记录。把：
```python
def run_chat_loop(chain: RunnableWithMessageHistory) -> None:
    print("LangChain CLI chatbot (with memory)")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        try:
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default"}},
            )
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Bot: {answer}")
```
改为：
```python
def run_chat_loop(chain: RunnableWithMessageHistory) -> None:
    print("LangChain CLI chatbot (with memory)")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        append_message("human", question)

        try:
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default"}},
            )
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        append_message("ai", answer)
        print(f"Bot: {answer}")
```

完整修改后的文件结构如下：
```python
from langchain_core.runnables.history import RunnableWithMessageHistory

from chatbot.config import ConfigError, load_config
from chatbot.history import append_message, load_history
from chatbot.llm import build_chain, build_llm, init_session_history


def run_chat_loop(chain: RunnableWithMessageHistory) -> None:
    print("LangChain CLI chatbot (with memory)")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        append_message("human", question)

        try:
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default"}},
            )
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        append_message("ai", answer)
        print(f"Bot: {answer}")


def main(argv=None) -> int:
    try:
        config = load_config(argv)
        records = load_history()
        llm = build_llm(config)
        init_session_history("default", records)
        chain = build_chain(llm)
        run_chat_loop(chain)
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

- [ ] **Step 2: 验证语法正确**

Run: `.venv/bin/python -m py_compile chatbot/main.py`
Expected: exit code 0, no output

- [ ] **Step 3: 验证所有测试通过**

Run: `.venv/bin/python -m pytest -v`
Expected: all 9 tests PASS

- [ ] **Step 4: Commit**

```bash
git add chatbot/main.py
git commit -m "feat: load JSON history on startup and append each turn"
```

### Task 4: 手动验证端到端行为

**No files changed.**

- [ ] **Step 1: 确保 `.env` 中有有效 API Key**

确认 `OPENAI_API_KEY` 已设置。

- [ ] **Step 2: 删除可能存在的旧历史文件**

```bash
rm -f chat_history.json
```

- [ ] **Step 3: 启动并运行多轮对话**

```bash
.venv/bin/python -m chatbot.main
```

输入：
1. `我的名字是张三`
2. `我叫什么名字？` — 期望回答 "张三"
3. `exit`

- [ ] **Step 4: 检查 JSON 文件**

```bash
cat chat_history.json
```

预期输出包含 3 条 human + 2 条 ai 记录，格式正确。

- [ ] **Step 5: 重启并验证跨会话记忆**

```bash
.venv/bin/python -m chatbot.main
```

输入：`我们之前聊过什么？` — 期望模型能复述上轮对话内容

- [ ] **Step 6: 删除 JSON 文件后启动验证**

```bash
rm chat_history.json && .venv/bin/python -m chatbot.main
```

输入 `exit` 退出，检查 `chat_history.json` 重新生成。

- [ ] **Step 7: 验证损坏文件不会阻塞启动**

```bash
echo "corrupted" > chat_history.json && .venv/bin/python -m chatbot.main
```

程序不应报错，正常启动。输入 `exit` 退出。清除损坏文件：
```bash
rm chat_history.json
```
