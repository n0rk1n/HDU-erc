# AI Message Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one-time like and dislike feedback for newly generated AI messages in the Web chat page.

**Architecture:** Store feedback metadata directly on new AI records in `data/records/chat_history.json`. Keep `chatbot/history.py` as the persistence boundary, expose a small FastAPI feedback endpoint in `chatbot/web.py`, return the generated AI message ID from SSE `done`, and let `chatbot/static/app.js` render controls only for unrated AI messages with IDs.

**Tech Stack:** Python, FastAPI, pytest, vanilla JavaScript, SSE, JSON file persistence.

---

## File Structure

- Modify: `chatbot/history.py`

  Owns history loading, history saving, AI message ID creation, AI message append behavior, and one-time feedback recording.

- Modify: `chatbot/chat_service.py`

  Writes AI replies through the new AI-specific history function and emits `message_id` in successful stream completion events.

- Modify: `chatbot/web.py`

  Returns `id` and `feedback` in session/history payloads and exposes `POST /api/messages/{message_id}/feedback`.

- Modify: `chatbot/static/app.js`

  Renders feedback buttons for unrated AI messages, submits feedback, and hides controls after success.

- Modify: `chatbot/static/style.css`

  Styles compact feedback controls and inline feedback failure text.

- Modify: `tests/test_history.py`

  Covers AI history records and feedback persistence states.

- Modify: `tests/test_chat_service.py`

  Covers `message_id` in stream completion events and non-persistence on stream failure.

- Modify: `tests/test_web.py`

  Covers API behavior, session payload metadata, and frontend rendering behavior through the existing Node VM pattern.

---

### Task 1: Add History Feedback Primitives

**Files:**

- Modify: `tests/test_history.py`

- Modify: `chatbot/history.py`

- [ ] **Step 1: Write failing history tests**

Add these imports near the top of `tests/test_history.py`:

```python
from chatbot.history import (
    append_ai_message,
    append_message,
    format_recent,
    load_history,
    record_message_feedback,
)
```

Add these tests after `test_append_message_appends`:

```python
def test_append_ai_message_creates_feedback_ready_record(history_file):
    record = append_ai_message("reply")

    data = json.loads(history_file.read_text())

    assert record == data[0]
    assert data[0]["role"] == "ai"
    assert data[0]["content"] == "reply"
    assert data[0]["id"].startswith("ai_")
    assert data[0]["feedback"] is None
    assert "timestamp" in data[0]


def test_append_message_keeps_human_record_shape(history_file):
    append_message("human", "hello")

    data = json.loads(history_file.read_text())

    assert data[0]["role"] == "human"
    assert data[0]["content"] == "hello"
    assert "timestamp" in data[0]
    assert "id" not in data[0]
    assert "feedback" not in data[0]
```

Add these tests after `test_load_history_returns_all_records`:

```python
def test_record_message_feedback_writes_first_rating(history_file):
    record = append_ai_message("reply")

    result = record_message_feedback(record["id"], "like")
    data = json.loads(history_file.read_text())

    assert result.status == "updated"
    assert result.feedback == "like"
    assert data[0]["feedback"] == "like"


def test_record_message_feedback_rejects_second_rating(history_file):
    record = append_ai_message("reply")
    record_message_feedback(record["id"], "like")

    result = record_message_feedback(record["id"], "dislike")
    data = json.loads(history_file.read_text())

    assert result.status == "already_rated"
    assert result.feedback == "like"
    assert data[0]["feedback"] == "like"


def test_record_message_feedback_returns_not_found(history_file):
    result = record_message_feedback("ai_missing", "like")

    assert result.status == "not_found"
    assert result.feedback == ""


def test_record_message_feedback_rejects_human_message(history_file):
    append_message("human", "hello")
    data = json.loads(history_file.read_text())
    data[0]["id"] = "human_1"
    history_file.write_text(json.dumps(data))

    result = record_message_feedback("human_1", "like")

    assert result.status == "not_ai"
    assert result.feedback == ""


def test_record_message_feedback_rejects_invalid_value(history_file):
    record = append_ai_message("reply")

    result = record_message_feedback(record["id"], "neutral")

    assert result.status == "invalid_feedback"
    assert result.feedback == ""
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
python -m pytest tests/test_history.py::test_append_ai_message_creates_feedback_ready_record tests/test_history.py::test_record_message_feedback_writes_first_rating -v
```

Expected: FAIL with an import error for `append_ai_message` or `record_message_feedback`.

- [ ] **Step 3: Implement history primitives**

In `chatbot/history.py`, add these imports:

```python
from dataclasses import dataclass
from uuid import uuid4
```

Add this dataclass after the module constants:

```python
@dataclass(frozen=True)
class FeedbackUpdateResult:
    status: str
    feedback: str = ""
```

Add these helpers before `append_message`:

```python
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save_history(records: list[dict]) -> bool:
    path = Path(HISTORY_FILE)
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return True
    except OSError as exc:
        print(f"Warning: could not save chat history: {exc}")
        return False
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError as exc:
            print(f"Warning: could not clean temporary history file: {exc}")
```

Replace the body of `append_message` with:

```python
def append_message(role: str, content: str) -> dict | None:
    """追加一条消息到历史文件 —— 先写临时文件再原子替换，避免写坏 JSON。"""
    try:
        records = load_history()
        record = {
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
        }
        records.append(record)
        if _save_history(records):
            return record
        return None
    except OSError as exc:
        print(f"Warning: could not save chat history: {exc}")
        return None
```

Add these functions after `append_message`:

```python
def append_ai_message(content: str) -> dict | None:
    try:
        records = load_history()
        record = {
            "id": f"ai_{uuid4().hex}",
            "role": "ai",
            "content": content,
            "timestamp": _now_iso(),
            "feedback": None,
        }
        records.append(record)
        if _save_history(records):
            return record
        return None
    except OSError as exc:
        print(f"Warning: could not save chat history: {exc}")
        return None


def record_message_feedback(message_id: str, feedback: str) -> FeedbackUpdateResult:
    if feedback not in {"like", "dislike"}:
        return FeedbackUpdateResult("invalid_feedback")

    records = load_history()
    for record in records:
        if record.get("id") != message_id:
            continue
        if record.get("role") != "ai":
            return FeedbackUpdateResult("not_ai")
        existing_feedback = record.get("feedback")
        if existing_feedback in {"like", "dislike"}:
            return FeedbackUpdateResult("already_rated", existing_feedback)
        record["feedback"] = feedback
        if _save_history(records):
            return FeedbackUpdateResult("updated", feedback)
        return FeedbackUpdateResult("write_failed")

    return FeedbackUpdateResult("not_found")
```

- [ ] **Step 4: Run history tests**

Run:

```bash
python -m pytest tests/test_history.py -v
```

Expected: PASS for all `tests/test_history.py` tests.

- [ ] **Step 5: Commit history primitives**

Run:

```bash
git add chatbot/history.py tests/test_history.py
git commit -m "feat: add ai message feedback persistence"
```

---

### Task 2: Add Feedback Web API And Session Metadata

**Files:**

- Modify: `tests/test_web.py`

- Modify: `chatbot/web.py`

- [ ] **Step 1: Write failing Web API tests**

Add this import near the top of `tests/test_web.py`:

```python
from chatbot.history import FeedbackUpdateResult
```

Add this test after `test_session_endpoint_returns_null_emotion`:

```python
def test_session_endpoint_preserves_message_feedback_metadata(monkeypatch):
    records = [
        {"role": "ai", "content": "old", "timestamp": "t1"},
        {
            "id": "ai_1",
            "role": "ai",
            "content": "new",
            "timestamp": "t2",
            "feedback": None,
        },
        {
            "id": "ai_2",
            "role": "ai",
            "content": "rated",
            "timestamp": "t3",
            "feedback": "like",
        },
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr("chatbot.web.load_analysis_records", lambda: [])

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json()["messages"] == records
```

Add these tests before `test_stream_endpoint_returns_sse_events`:

```python
def test_feedback_endpoint_records_like(monkeypatch):
    captured = {}

    def fake_record_feedback(message_id, feedback):
        captured["message_id"] = message_id
        captured["feedback"] = feedback
        return FeedbackUpdateResult("updated", feedback)

    monkeypatch.setattr("chatbot.web.record_message_feedback", fake_record_feedback)

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.post("/api/messages/ai_1/feedback", json={"feedback": "like"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "updated",
        "message_id": "ai_1",
        "feedback": "like",
    }
    assert captured == {"message_id": "ai_1", "feedback": "like"}


def test_feedback_endpoint_records_already_rated(monkeypatch):
    monkeypatch.setattr(
        "chatbot.web.record_message_feedback",
        lambda message_id, feedback: FeedbackUpdateResult("already_rated", "dislike"),
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.post("/api/messages/ai_1/feedback", json={"feedback": "like"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "already_rated",
        "message_id": "ai_1",
        "feedback": "dislike",
    }


def test_feedback_endpoint_returns_not_found(monkeypatch):
    monkeypatch.setattr(
        "chatbot.web.record_message_feedback",
        lambda message_id, feedback: FeedbackUpdateResult("not_found"),
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.post("/api/messages/ai_missing/feedback", json={"feedback": "like"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Message not found."}


def test_feedback_endpoint_rejects_non_ai_message(monkeypatch):
    monkeypatch.setattr(
        "chatbot.web.record_message_feedback",
        lambda message_id, feedback: FeedbackUpdateResult("not_ai"),
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.post("/api/messages/human_1/feedback", json={"feedback": "like"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Feedback is only supported for AI messages."}


def test_feedback_endpoint_rejects_invalid_feedback():
    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.post("/api/messages/ai_1/feedback", json={"feedback": "neutral"})

    assert response.status_code == 422


def test_feedback_endpoint_returns_write_failure(monkeypatch):
    monkeypatch.setattr(
        "chatbot.web.record_message_feedback",
        lambda message_id, feedback: FeedbackUpdateResult("write_failed"),
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.post("/api/messages/ai_1/feedback", json={"feedback": "like"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not save feedback."}
```

- [ ] **Step 2: Run the new Web API tests and verify they fail**

Run:

```bash
python -m pytest tests/test_web.py::test_feedback_endpoint_records_like tests/test_web.py::test_session_endpoint_preserves_message_feedback_metadata -v
```

Expected: FAIL because `record_message_feedback` is not imported in `chatbot.web` and the endpoint is missing.

- [ ] **Step 3: Implement Web API changes**

In `chatbot/web.py`, update imports:

```python
from typing import Literal

from pydantic import BaseModel

from chatbot.history import load_history, record_message_feedback
```

Add this model after `STATIC_DIR`:

```python
class FeedbackRequest(BaseModel):
    feedback: Literal["like", "dislike"]
```

Replace `_structured_messages` with:

```python
def _structured_messages(records: list[dict], limit: int) -> list[dict]:
    messages = []
    for record in records:
        if record.get("role") not in {"human", "ai"}:
            continue
        message = {
            "role": record.get("role", ""),
            "content": record.get("content", ""),
            "timestamp": record.get("timestamp", ""),
        }
        if "id" in record:
            message["id"] = record.get("id")
        if "feedback" in record:
            message["feedback"] = record.get("feedback")
        messages.append(message)
    return messages[-limit:]
```

Add this endpoint after `session`:

```python
    @app.post("/api/messages/{message_id}/feedback")
    def message_feedback(message_id: str, request: FeedbackRequest):
        result = record_message_feedback(message_id, request.feedback)
        if result.status in {"updated", "already_rated"}:
            return {
                "status": result.status,
                "message_id": message_id,
                "feedback": result.feedback,
            }
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Message not found.")
        if result.status == "not_ai":
            raise HTTPException(
                status_code=400,
                detail="Feedback is only supported for AI messages.",
            )
        if result.status == "write_failed":
            raise HTTPException(status_code=500, detail="Could not save feedback.")
        raise HTTPException(status_code=400, detail="Invalid feedback.")
```

- [ ] **Step 4: Run Web API tests**

Run:

```bash
python -m pytest tests/test_web.py::test_session_endpoint_preserves_message_feedback_metadata tests/test_web.py::test_feedback_endpoint_records_like tests/test_web.py::test_feedback_endpoint_records_already_rated tests/test_web.py::test_feedback_endpoint_returns_not_found tests/test_web.py::test_feedback_endpoint_rejects_non_ai_message tests/test_web.py::test_feedback_endpoint_rejects_invalid_feedback tests/test_web.py::test_feedback_endpoint_returns_write_failure -v
```

Expected: PASS for all listed tests.

- [ ] **Step 5: Commit Web API changes**

Run:

```bash
git add chatbot/web.py tests/test_web.py
git commit -m "feat: add message feedback endpoint"
```

---

### Task 3: Emit Message IDs From ChatService SSE

**Files:**

- Modify: `tests/test_chat_service.py`

- Modify: `chatbot/chat_service.py`

- Modify: `tests/test_web.py`

- [ ] **Step 1: Write failing ChatService tests**

In `tests/test_chat_service.py`, update `test_stream_reply_falls_back_to_invoke_and_writes_ai_message` so the monkeypatches and assertions are:

```python
    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"id": "ai_1", "role": "ai", "content": content, "feedback": None},
    )
```

```python
    assert [(event.event, event.data) for event in events] == [
        ("user_message", {"role": "human", "content": "hello"}),
        ("token", {"content": "full reply"}),
        ("done", {"content": "full reply", "message_id": "ai_1"}),
    ]
    assert stored_messages == [("human", "hello")]
```

In `tests/test_chat_service.py`, update `test_stream_reply_emits_user_tokens_and_done` with the same `append_ai_message` monkeypatch and change the final assertion to:

```python
    assert (done.event, done.data) == (
        "done",
        {"content": "hello world", "message_id": "ai_1"},
    )
    assert stored_messages == [("human", "hello")]
```

Add this test after `test_stream_reply_emits_user_tokens_and_done`:

```python
def test_stream_reply_records_ai_session_message_with_metadata(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "timestamp": "t1",
            "feedback": None,
        },
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert events[-1].data == {"content": "hello world", "message_id": "ai_1"}
    assert service.session_records[-1] == {
        "id": "ai_1",
        "role": "ai",
        "content": "hello world",
        "timestamp": "t1",
        "feedback": None,
    }
```

In `tests/test_web.py`, update `FakeService.stream_reply` so its final event is:

```python
        yield ChatEvent("done", {"content": "hi", "message_id": "ai_1"})
```

Update `test_stream_endpoint_returns_sse_events` to assert:

```python
    assert 'data: {"content": "hi", "message_id": "ai_1"}' in response.text
```

- [ ] **Step 2: Run the changed ChatService tests and verify they fail**

Run:

```bash
python -m pytest tests/test_chat_service.py::test_stream_reply_emits_user_tokens_and_done tests/test_chat_service.py::test_stream_reply_records_ai_session_message_with_metadata -v
```

Expected: FAIL because `append_ai_message` is not imported in `chatbot.chat_service` and `done` does not include `message_id`.

- [ ] **Step 3: Implement ChatService message ID emission**

In `chatbot/chat_service.py`, update the history import:

```python
from chatbot.history import append_ai_message, append_message
```

Add this method inside `ChatService` after `_append_user_message`:

```python
    def _append_ai_message(self, answer: str) -> str | None:
        record = append_ai_message(answer)
        if record is None:
            record = {"role": "ai", "content": answer}
        self.session_records.append(record)
        message_id = record.get("id")
        if isinstance(message_id, str) and message_id:
            return message_id
        return None
```

In `generate_reply`, replace:

```python
        append_message("ai", answer)
        self.session_records.append({"role": "ai", "content": answer})
```

with:

```python
        self._append_ai_message(answer)
```

In `stream_reply`, replace:

```python
        append_message("ai", answer)
        self.session_records.append({"role": "ai", "content": answer})
        yield ChatEvent("done", {"content": answer})
```

with:

```python
        message_id = self._append_ai_message(answer)
        data = {"content": answer}
        if message_id:
            data["message_id"] = message_id
        yield ChatEvent("done", data)
```

- [ ] **Step 4: Run ChatService and stream endpoint tests**

Run:

```bash
python -m pytest tests/test_chat_service.py tests/test_web.py::test_stream_endpoint_returns_sse_events -v
```

Expected: PASS for all listed tests.

- [ ] **Step 5: Commit ChatService SSE changes**

Run:

```bash
git add chatbot/chat_service.py tests/test_chat_service.py tests/test_web.py
git commit -m "feat: return ai message id from stream"
```

---

### Task 4: Add Frontend Feedback Controls

**Files:**

- Modify: `tests/test_web.py`

- Modify: `chatbot/static/app.js`

- Modify: `chatbot/static/style.css`

- [ ] **Step 1: Write failing frontend behavior test**

In `tests/test_web.py`, add this test after `test_static_app_js_initializes_from_session_snapshot`:

```python
def test_static_app_js_renders_and_submits_feedback_controls():
    root = Path(__file__).resolve().parents[1]
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for app.js behavior test")

    script = r"""
const fs = require("fs");
const vm = require("vm");

class Element {
  constructor(name) {
    this.name = name;
    this.children = [];
    this.parent = null;
    this.textContent = "";
    this.className = "";
    this.disabled = false;
    this.value = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.listeners = {};
    this.attributes = {};
  }

  appendChild(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  remove() {
    if (!this.parent) {
      return;
    }
    this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }

  requestSubmit() {}

  focus() {}

  set innerHTML(value) {
    this.children = [];
    this._innerHTML = value;
  }

  get innerHTML() {
    return this._innerHTML || "";
  }
}

const messagesEl = new Element("messages");
const formEl = new Element("form");
const inputEl = new Element("input");
const buttonEl = new Element("button");
const emotionStatusEl = new Element("emotion");
const fetchCalls = [];

const elements = {
  "#messages": messagesEl,
  "#chat-form": formEl,
  "#message-input": inputEl,
  "#send-button": buttonEl,
  "#emotion-status": emotionStatusEl,
};

const context = {
  console,
  encodeURIComponent,
  EventSource: function EventSource() {},
  fetch: async (url, options) => {
    fetchCalls.push({url, options});
    if (url === "/api/session?limit=10") {
      return {
        ok: true,
        json: async () => ({
          messages: [
            {role: "ai", content: "old"},
            {role: "ai", content: "new", id: "ai_1", feedback: null},
            {role: "ai", content: "rated", id: "ai_2", feedback: "like"},
          ],
          emotion: null,
        }),
      };
    }
    if (url === "/api/messages/ai_1/feedback") {
      return {
        ok: true,
        json: async () => ({status: "updated", message_id: "ai_1", feedback: "like"}),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  },
  document: {
    querySelector: (selector) => elements[selector],
    createElement: (name) => new Element(name),
  },
};

context.EventSource.prototype.addEventListener = function addEventListener() {};
context.EventSource.prototype.close = function close() {};

const code = fs.readFileSync("chatbot/static/app.js", "utf-8");
vm.runInNewContext(code, context);

setImmediate(async () => {
  try {
    if (messagesEl.children.length !== 3) {
      throw new Error(`expected 3 messages, got ${messagesEl.children.length}`);
    }
    if (messagesEl.children[0].children.length !== 1) {
      throw new Error("old AI message should not show feedback controls");
    }
    if (messagesEl.children[1].children.length !== 2) {
      throw new Error("new AI message should show feedback controls");
    }
    if (messagesEl.children[2].children.length !== 1) {
      throw new Error("rated AI message should not show feedback controls");
    }

    const controls = messagesEl.children[1].children[1];
    const likeButton = controls.children[0];
    if (likeButton.textContent !== "赞") {
      throw new Error(`unexpected like button text: ${likeButton.textContent}`);
    }

    await likeButton.listeners.click();

    if (fetchCalls[1].url !== "/api/messages/ai_1/feedback") {
      throw new Error(`unexpected feedback url: ${fetchCalls[1].url}`);
    }
    if (fetchCalls[1].options.method !== "POST") {
      throw new Error(`unexpected feedback method: ${fetchCalls[1].options.method}`);
    }
    if (fetchCalls[1].options.body !== JSON.stringify({feedback: "like"})) {
      throw new Error(`unexpected feedback body: ${fetchCalls[1].options.body}`);
    }
    if (messagesEl.children[1].children.length !== 1) {
      throw new Error("feedback controls should be removed after successful rating");
    }
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
});
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the frontend test and verify it fails**

Run:

```bash
python -m pytest tests/test_web.py::test_static_app_js_renders_and_submits_feedback_controls -v
```

Expected: FAIL because AI feedback controls are not rendered.

- [ ] **Step 3: Implement frontend feedback rendering**

In `chatbot/static/app.js`, replace `addMessage` with:

```javascript
function addMessage(role, content = "", metadata = {}) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.appendChild(bubble);
  if (role === "ai") {
    renderFeedbackControls(wrapper, metadata);
  }
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return {wrapper, bubble};
}
```

Add these functions after `addMessage`:

```javascript
function shouldShowFeedback(metadata) {
  return Boolean(metadata.id) && !metadata.feedback;
}

function renderFeedbackControls(wrapper, metadata) {
  if (!shouldShowFeedback(metadata)) {
    return;
  }

  const controls = document.createElement("div");
  controls.className = "feedback-controls";

  const likeButton = document.createElement("button");
  likeButton.type = "button";
  likeButton.className = "feedback-button";
  likeButton.textContent = "赞";
  likeButton.setAttribute("aria-label", "点赞");

  const dislikeButton = document.createElement("button");
  dislikeButton.type = "button";
  dislikeButton.className = "feedback-button";
  dislikeButton.textContent = "踩";
  dislikeButton.setAttribute("aria-label", "点踩");

  const status = document.createElement("span");
  status.className = "feedback-status";

  likeButton.addEventListener("click", () => submitFeedback(metadata.id, "like", controls, status));
  dislikeButton.addEventListener("click", () => submitFeedback(metadata.id, "dislike", controls, status));

  controls.appendChild(likeButton);
  controls.appendChild(dislikeButton);
  controls.appendChild(status);
  wrapper.appendChild(controls);
}

async function submitFeedback(messageId, feedback, controls, status) {
  const buttons = controls.children;
  Array.from(buttons).forEach((button) => {
    if (button.tagName === "BUTTON" || button.name === "button") {
      button.disabled = true;
    }
  });
  status.textContent = "";

  try {
    const response = await fetch(`/api/messages/${encodeURIComponent(messageId)}/feedback`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({feedback}),
    });
    if (!response.ok) {
      throw new Error("failed");
    }
    await response.json();
    controls.remove();
  } catch (error) {
    Array.from(buttons).forEach((button) => {
      if (button.tagName === "BUTTON" || button.name === "button") {
        button.disabled = false;
      }
    });
    status.textContent = "评价保存失败";
  }
}
```

Update `loadSession` message rendering:

```javascript
  payload.messages.forEach((message) => {
    const role = message.role === "human" ? "human" : "ai";
    addMessage(role, message.content, message);
  });
```

Update `streamMessage` state and token handling:

```javascript
  let aiMessage = null;
```

```javascript
    if (!aiMessage) {
      aiMessage = addMessage("ai", "");
    }
    aiMessage.bubble.textContent += payload.content;
```

Update `done` handling:

```javascript
  source.addEventListener("done", (event) => {
    const payload = JSON.parse(event.data);
    if (aiMessage && payload.message_id) {
      renderFeedbackControls(aiMessage.wrapper, {
        id: payload.message_id,
        feedback: null,
      });
    }
    source.close();
    setLocked(false);
    inputEl.focus();
  });
```

Update `error` handling:

```javascript
    if (!aiMessage) {
      aiMessage = addMessage("ai", "");
    }
    if (event.data) {
      const payload = JSON.parse(event.data);
      aiMessage.bubble.textContent = payload.message;
    } else {
      aiMessage.bubble.textContent = "连接中断，请稍后重试";
    }
```

- [ ] **Step 4: Implement feedback styles**

Add this CSS after `.ai .bubble` in `chatbot/static/style.css`:

```css
.message.ai {
  align-items: flex-start;
  flex-direction: column;
}

.feedback-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
}

.feedback-button {
  min-width: 36px;
  min-height: 32px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
  color: var(--muted);
  font-size: 13px;
}

.feedback-button:hover:not(:disabled) {
  border-color: var(--human);
  color: var(--human);
}

.feedback-status {
  color: var(--muted);
  font-size: 13px;
}
```

- [ ] **Step 5: Run frontend tests**

Run:

```bash
python -m pytest tests/test_web.py::test_static_app_js_initializes_from_session_snapshot tests/test_web.py::test_static_app_js_renders_and_submits_feedback_controls -v
```

Expected: PASS for both frontend VM tests.

- [ ] **Step 6: Commit frontend controls**

Run:

```bash
git add chatbot/static/app.js chatbot/static/style.css tests/test_web.py
git commit -m "feat: add ai feedback controls"
```

---

### Task 5: Full Verification

**Files:**

- Verify: `chatbot/history.py`

- Verify: `chatbot/chat_service.py`

- Verify: `chatbot/web.py`

- Verify: `chatbot/static/app.js`

- Verify: `chatbot/static/style.css`

- Verify: `tests/test_history.py`

- Verify: `tests/test_chat_service.py`

- Verify: `tests/test_web.py`

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python -m pytest -v
```

Expected: PASS for the full test suite.

- [ ] **Step 2: Inspect the final diff**

Run:

```bash
git diff --stat HEAD~4..HEAD
```

Expected: The diff includes only feedback-related changes in the files listed in this plan.

- [ ] **Step 3: Review JSON field behavior manually**

Run:

```bash
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import chatbot.history as history

with TemporaryDirectory() as directory:
    path = Path(directory) / "chat_history.json"
    original = history.HISTORY_FILE
    history.HISTORY_FILE = str(path)
    try:
        human = history.append_message("human", "hello")
        ai = history.append_ai_message("hi")
        result = history.record_message_feedback(ai["id"], "like")
        records = json.loads(path.read_text())
        print(human)
        print(records)
        print(result)
    finally:
        history.HISTORY_FILE = original
PY
```

Expected: The printed human record has no `id` or `feedback`; the printed AI record has an `ai_` ID and `feedback` set to `like`; the printed result status is `updated`.

- [ ] **Step 4: Confirm worktree state**

Run:

```bash
git status --short
```

Expected: No feedback implementation files are unstaged. Pre-existing unrelated changes may remain if they were present before execution.
