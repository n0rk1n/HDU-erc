# 单次运行对话记忆实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 CLI 聊天机器人加入 `RunnableWithMessageHistory` + `InMemoryChatMessageHistory`，实现单次运行内的对话记忆。

**Architecture:** 在 `chatbot/llm.py` 中新增 `build_chain(llm)` 函数，使用 LCEL（`prompt | llm`）构建链，用 `RunnableWithMessageHistory` 包装并传入 `InMemoryChatMessageHistory` 管理对话历史。`chatbot/main.py` 的 `run_chat_loop` 入参从 `ChatOpenAI` 改为 `RunnableWithMessageHistory`，循环内用 `chain.invoke()` 替代 `ask_once()`。每轮对话通过 `session_id="default"` 自动读取和追加历史。

**Tech Stack:** Python 3.13, LangChain 1.3.0, langchain-core 1.4.0, RunnableWithMessageHistory, InMemoryChatMessageHistory

---

### Task 1: 修改 `chatbot/llm.py` — 新增 `build_chain()` 函数

**Files:**
- Modify: `chatbot/llm.py`

- [ ] **Step 1: 替换 import 并新增 `build_chain()` 函数**

移除旧 import（不在 LangChain 1.3.0 中存在）：
```python
# 删除以下两行
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
```

添加新 import：
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
```

在文件末尾追加以下新代码：
```python
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chain(llm: ChatOpenAI) -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
```

- [ ] **Step 2: 验证语法正确**

Run: `.venv/bin/python -m py_compile chatbot/llm.py`
Expected: exit code 0, no output

- [ ] **Step 3: Commit**

```bash
git add chatbot/llm.py
git commit -m "feat: add build_chain with RunnableWithMessageHistory"
```

### Task 2: 修改 `chatbot/main.py` — 接入带记忆的链

**Files:**
- Modify: `chatbot/main.py`

- [ ] **Step 1: 替换 import 并更新 `run_chat_loop` 和 `main()`**

修改 import 行。把：
```python
from chatbot.llm import ask_once, build_llm
```
改为：
```python
from chatbot.llm import build_chain, build_llm
```

移除不再使用的 import（`ChatOpenAI` 不再需要做类型标注）：
```python
# 删除: from langchain_openai import ChatOpenAI
```

添加新 import：
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
```

修改 `run_chat_loop` 签名和内部调用。把：
```python
def run_chat_loop(llm: ChatOpenAI) -> None:
    print("LangChain CLI chatbot")
    ...
        answer = ask_once(llm, question)
```
改为：
```python
def run_chat_loop(chain: RunnableWithMessageHistory) -> None:
    print("LangChain CLI chatbot (with memory)")
    ...
        answer = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "default"}},
        )
```

修改 `main()` 中的调用顺序。把：
```python
        llm = build_llm(config)
        run_chat_loop(llm)
```
改为：
```python
        llm = build_llm(config)
        chain = build_chain(llm)
        run_chat_loop(chain)
```

**注意：** `RunnableWithMessageHistory.invoke()` 返回的是一个 `AIMessage` 对象（或完整 `dict`），不是纯字符串。需要提取其 `content` 属性。确认 `answer.content` 或 `answer["content"]` 的正确提取方式。

完整修改后的文件结构如下：
```python
from langchain_core.runnables.history import RunnableWithMessageHistory

from chatbot.config import ConfigError, load_config
from chatbot.llm import build_chain, build_llm


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


def main(argv=None) -> int:
    try:
        config = load_config(argv)
        llm = build_llm(config)
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

- [ ] **Step 3: 验证现有测试仍然通过**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests PASS

- [ ] **Step 4: 手动验证记忆功能**

确认 `.env` 中有有效的 `OPENAI_API_KEY`，然后：
```
.venv/bin/python -m chatbot.main
```
启动后输入以下对话，验证模型能引用上下文：
1. 输入：`我的名字是张三`
2. 输入：`我叫什么名字？` — 期望回答 "张三"

- [ ] **Step 5: Commit**

```bash
git add chatbot/main.py
git commit -m "feat: integrate RunnableWithMessageHistory into chat loop"
```
