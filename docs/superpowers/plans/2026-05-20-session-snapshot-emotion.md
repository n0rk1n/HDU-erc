# Session Snapshot Emotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the previous successful emotion state on Web startup, return it with recent messages from one session snapshot endpoint, and use it for the next chat payload.

**Architecture:** Keep chat history and emotion analysis in separate JSON files. Add a focused emotion-store reader, pass the latest successful emotion into `ChatService`, expose `GET /api/session?limit=10`, and update the browser initializer to render messages and emotion from that single response.

**Tech Stack:** Python, FastAPI, pytest, vanilla JavaScript, existing JSON persistence.

---

## File Structure

- Modify `chatbot/emotion.py`

  Add project-root anchoring for `EMOTION_ANALYSIS_FILE` and a `load_latest_successful_emotion()` helper that returns a small dictionary or `None`.

- Modify `tests/test_emotion.py`

  Cover path anchoring and latest successful emotion selection.

- Modify `chatbot/chat_service.py`

  Accept `initial_emotion` in the constructor and initialize `current_emotion` from it.

- Modify `tests/test_chat_service.py`

  Verify the first runtime payload includes the loaded emotion context.

- Modify `chatbot/web.py`

  Import `load_latest_successful_emotion()`, pass the initial emotion into `ChatService`, add `_session_snapshot()`, and expose `GET /api/session`.

- Modify `tests/test_web.py`

  Cover `build_service()` emotion initialization and session snapshot response.

- Modify `chatbot/static/app.js`

  Replace startup history loading with session snapshot loading and initial emotion rendering.

- Modify `tests/test_web.py`

  Add a source-level check that `app.js` calls `/api/session?limit=10` and no longer calls `/api/history?limit=10` during startup.

---

### Task 1: Emotion Store Latest Successful Reader

**Files:**

- Modify: `chatbot/emotion.py`

- Test: `tests/test_emotion.py`

- [ ] **Step 1: Write failing tests for default path anchoring and latest emotion lookup**

Append these tests to `tests/test_emotion.py`:

```python
from pathlib import Path

import chatbot.emotion as emotion
```

If `Path` or `emotion` imports already exist after editing, keep a single import for each.

Then append:

```python
def test_default_emotion_analysis_file_is_project_data_file():
    path = Path(emotion.EMOTION_ANALYSIS_FILE)

    assert path.is_absolute()
    assert path.name == "emotion_analysis.json"
    assert path.parent.name == "data"
    assert path.parent.parent == Path(__file__).resolve().parents[1]


def test_load_latest_successful_emotion_returns_latest_success(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text(json.dumps([
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "anxious",
            "success": True,
        },
        {
            "timestamp": "2026-05-19T18:10:00+08:00",
            "turn_count": 4,
            "emotion": "",
            "success": False,
            "error": "Failed to parse a known emotion label.",
        },
        {
            "timestamp": "2026-05-19T18:20:00+08:00",
            "turn_count": 5,
            "emotion": "sad",
            "success": True,
        },
    ]), encoding="utf-8")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "sad",
        "timestamp": "2026-05-19T18:20:00+08:00",
        "turn_count": 5,
    }


def test_load_latest_successful_emotion_skips_trailing_failures(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text(json.dumps([
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "anxious",
            "success": True,
        },
        {
            "timestamp": "2026-05-19T18:10:00+08:00",
            "turn_count": 4,
            "emotion": "",
            "success": False,
        },
    ]), encoding="utf-8")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() == {
        "emotion": "anxious",
        "timestamp": "2026-05-19T18:00:00+08:00",
        "turn_count": 3,
    }


def test_load_latest_successful_emotion_returns_none_without_success(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    analysis_file.write_text(json.dumps([
        {
            "timestamp": "2026-05-19T18:00:00+08:00",
            "turn_count": 3,
            "emotion": "",
            "success": False,
        }
    ]), encoding="utf-8")
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() is None


def test_load_latest_successful_emotion_returns_none_for_missing_file(tmp_path, monkeypatch):
    analysis_file = tmp_path / "emotion_analysis.json"
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    assert emotion.load_latest_successful_emotion() is None
```

- [ ] **Step 2: Run the new emotion tests to verify they fail**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_emotion.py -q
```

Expected: FAIL because `load_latest_successful_emotion` does not exist, and the default emotion path is currently relative.

- [ ] **Step 3: Implement project-root path anchoring and latest successful lookup**

In `chatbot/emotion.py`, replace:

```python
EMOTION_ANALYSIS_FILE = "data/emotion_analysis.json"
```

with:

```python
EMOTION_ANALYSIS_FILE = str(Path(__file__).resolve().parents[1] / "data" / "emotion_analysis.json")
```

Then add this function after `load_analysis_records()`:

```python
def load_latest_successful_emotion() -> dict[str, Any] | None:
    records = load_analysis_records()
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        emotion = str(record.get("emotion", "")).strip()
        if not record.get("success") or not emotion:
            continue
        return {
            "emotion": emotion,
            "timestamp": record.get("timestamp", ""),
            "turn_count": record.get("turn_count", 0),
        }
    return None
```

- [ ] **Step 4: Run emotion tests to verify they pass**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_emotion.py -q
```

Expected: PASS for all `tests/test_emotion.py` tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add chatbot/emotion.py tests/test_emotion.py
git commit -m "feat: load latest successful emotion"
```

---

### Task 2: ChatService Initial Emotion

**Files:**

- Modify: `chatbot/chat_service.py`

- Test: `tests/test_chat_service.py`

- [ ] **Step 1: Write failing test for initial emotion context**

Append this test to `tests/test_chat_service.py`:

```python
def test_generate_reply_uses_initial_emotion_context(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        initial_emotion="sad",
    )

    assert service.generate_reply("q1") == "reply 1"
    assert chain.payloads[0]["emotion_context"] == "Current detected user emotion: sad"
```

- [ ] **Step 2: Run the new ChatService test to verify it fails**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_chat_service.py::test_generate_reply_uses_initial_emotion_context -q
```

Expected: FAIL with `TypeError` because `ChatService.__init__()` does not accept `initial_emotion`.

- [ ] **Step 3: Implement `initial_emotion` in `ChatService`**

In `chatbot/chat_service.py`, change the constructor signature from:

```python
        initial_records: list[dict] | None = None,
        session_id: str = "default",
```

to:

```python
        initial_records: list[dict] | None = None,
        session_id: str = "default",
        initial_emotion: str = "",
```

Then replace:

```python
        self.current_emotion = ""
```

with:

```python
        self.current_emotion = initial_emotion
```

- [ ] **Step 4: Run ChatService tests**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_chat_service.py -q
```

Expected: PASS for all `tests/test_chat_service.py` tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add chatbot/chat_service.py tests/test_chat_service.py
git commit -m "feat: initialize chat service emotion"
```

---

### Task 3: Web Session Snapshot Endpoint

**Files:**

- Modify: `chatbot/web.py`

- Test: `tests/test_web.py`

- [ ] **Step 1: Write failing tests for service initialization and session response**

In `tests/test_web.py`, add these tests after `test_build_service_does_not_duplicate_session_history`:

```python
def test_build_service_uses_latest_successful_emotion(monkeypatch):
    records = [
        {"role": "human", "content": "hello"},
        {"role": "ai", "content": "hi"},
    ]

    class FakeLlm:
        pass

    captured = {}

    def fake_chat_service(chain, config, emotion_llm, initial_records=None, initial_emotion="", session_id="default"):
        captured["initial_records"] = initial_records
        captured["initial_emotion"] = initial_emotion
        return object()

    monkeypatch.setattr("chatbot.web.load_config", lambda argv: object())
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_latest_successful_emotion",
        lambda: {"emotion": "sad", "timestamp": "t1", "turn_count": 5},
    )
    monkeypatch.setattr("chatbot.web.load_profile", lambda: {})
    monkeypatch.setattr("chatbot.web.format_profile", lambda profile: "")
    monkeypatch.setattr(
        "chatbot.web.build_runtime_llms",
        lambda config: (FakeLlm(), FakeLlm()),
    )
    monkeypatch.setattr("chatbot.web.build_chain", lambda llm, profile_text: object())
    monkeypatch.setattr("chatbot.web.ChatService", fake_chat_service)

    build_service()

    assert captured["initial_records"] == records
    assert captured["initial_emotion"] == "sad"


def test_session_endpoint_returns_messages_and_latest_emotion(monkeypatch):
    records = [
        {"role": "human", "content": f"q{i}", "timestamp": f"t{i}"}
        for i in range(12)
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_latest_successful_emotion",
        lambda: {"emotion": "sad", "timestamp": "emotion-time", "turn_count": 5},
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["messages"][0] == {"role": "human", "content": "q2", "timestamp": "t2"}
    assert payload["messages"][-1] == {"role": "human", "content": "q11", "timestamp": "t11"}
    assert payload["emotion"] == {
        "emotion": "sad",
        "timestamp": "emotion-time",
        "turn_count": 5,
    }


def test_session_endpoint_returns_null_emotion(monkeypatch):
    monkeypatch.setattr("chatbot.web.load_history", lambda: [])
    monkeypatch.setattr("chatbot.web.load_latest_successful_emotion", lambda: None)

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json() == {"messages": [], "emotion": None}
```

- [ ] **Step 2: Run the new Web tests to verify they fail**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest \
  tests/test_web.py::test_build_service_uses_latest_successful_emotion \
  tests/test_web.py::test_session_endpoint_returns_messages_and_latest_emotion \
  tests/test_web.py::test_session_endpoint_returns_null_emotion \
  -q
```

Expected: FAIL because `chatbot.web.load_latest_successful_emotion` is not imported, `build_service()` does not pass `initial_emotion`, and `/api/session` does not exist.

- [ ] **Step 3: Import the emotion reader**

In `chatbot/web.py`, add this import near the other `chatbot.*` imports:

```python
from chatbot.emotion import load_latest_successful_emotion
```

- [ ] **Step 4: Pass initial emotion to `ChatService`**

In `build_service()` in `chatbot/web.py`, replace:

```python
    chat_llm, emotion_llm = build_runtime_llms(config)
    init_session_history("default", records)
    chain = build_chain(chat_llm, profile_text)
    return ChatService(chain, config, emotion_llm, initial_records=records)
```

with:

```python
    chat_llm, emotion_llm = build_runtime_llms(config)
    latest_emotion = load_latest_successful_emotion()
    init_session_history("default", records)
    chain = build_chain(chat_llm, profile_text)
    return ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=records,
        initial_emotion=(latest_emotion or {}).get("emotion", ""),
    )
```

- [ ] **Step 5: Add session snapshot helper and endpoint**

In `chatbot/web.py`, add this helper after `_recent_messages()`:

```python
def _session_snapshot(limit: int) -> dict:
    return {
        "messages": _recent_messages(limit),
        "emotion": load_latest_successful_emotion(),
    }
```

Then add this route after the existing `/api/history` route:

```python
    @app.get("/api/session")
    def session(limit: int = Query(default=10, gt=0, le=100)):
        return _session_snapshot(limit)
```

- [ ] **Step 6: Run Web tests**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_web.py -q
```

Expected: PASS for all `tests/test_web.py` tests.

- [ ] **Step 7: Commit**

Run:

```bash
git add chatbot/web.py tests/test_web.py
git commit -m "feat: add session snapshot endpoint"
```

---

### Task 4: Frontend Startup Snapshot Loading

**Files:**

- Modify: `chatbot/static/app.js`

- Test: `tests/test_web.py`

- [ ] **Step 1: Write failing source test for frontend startup endpoint**

Append this test to `tests/test_web.py`:

```python
def test_static_app_js_loads_session_snapshot():
    root = Path(__file__).resolve().parents[1]
    app_js = (root / "chatbot" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/session?limit=10")' in app_js
    assert 'fetch("/api/history?limit=10")' not in app_js
    assert "payload.emotion" in app_js
    assert "情感状态：暂无" in app_js
```

- [ ] **Step 2: Run the new frontend source test to verify it fails**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_web.py::test_static_app_js_loads_session_snapshot -q
```

Expected: FAIL because `app.js` still calls `/api/history?limit=10`.

- [ ] **Step 3: Replace `loadHistory()` with `loadSession()`**

In `chatbot/static/app.js`, replace the full `loadHistory()` function:

```javascript
async function loadHistory() {
  const response = await fetch("/api/history?limit=10");
  if (!response.ok) {
    emotionStatusEl.textContent = "历史加载失败";
    return;
  }
  const payload = await response.json();
  messagesEl.innerHTML = "";
  payload.messages.forEach((message) => {
    const role = message.role === "human" ? "human" : "ai";
    addMessage(role, message.content);
  });
}
```

with:

```javascript
function renderEmotion(emotion) {
  if (emotion && emotion.emotion) {
    emotionStatusEl.textContent = `情感状态：${emotion.emotion}`;
    return;
  }
  emotionStatusEl.textContent = "情感状态：暂无";
}

async function loadSession() {
  const response = await fetch("/api/session?limit=10");
  if (!response.ok) {
    emotionStatusEl.textContent = "历史加载失败";
    return;
  }
  const payload = await response.json();
  messagesEl.innerHTML = "";
  payload.messages.forEach((message) => {
    const role = message.role === "human" ? "human" : "ai";
    addMessage(role, message.content);
  });
  renderEmotion(payload.emotion);
}
```

- [ ] **Step 4: Update initializer**

In `chatbot/static/app.js`, replace:

```javascript
    await loadHistory();
```

with:

```javascript
    await loadSession();
```

- [ ] **Step 5: Run frontend source and Web tests**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_web.py -q
```

Expected: PASS for all `tests/test_web.py` tests.

- [ ] **Step 6: Commit**

Run:

```bash
git add chatbot/static/app.js tests/test_web.py
git commit -m "feat: load session snapshot on startup"
```

---

### Task 5: Full Verification

**Files:**

- Verify only.

- [ ] **Step 1: Run the focused test modules**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest tests/test_emotion.py tests/test_chat_service.py tests/test_web.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m pytest -q
```

Expected: PASS. FastAPI deprecation warnings from existing `@app.on_event("startup")` usage are acceptable if the tests pass.

- [ ] **Step 3: Start local Web service for smoke testing**

Run:

```bash
/private/tmp/hdu-chatbot-venv/bin/python -m uvicorn chatbot.web:app --host 127.0.0.1 --port 8080
```

Expected: server starts and prints a URL for `http://127.0.0.1:8080`.

- [ ] **Step 4: Smoke test the new session endpoint**

In another terminal, run:

```bash
curl -i --max-time 5 'http://127.0.0.1:8080/api/session?limit=10'
```

Expected: `HTTP/1.1 200 OK` and a JSON body containing both `messages` and `emotion`.

- [ ] **Step 5: Smoke test the browser page**

Open:

```text
http://127.0.0.1:8080/
```

Expected: the page renders recent messages. If `data/emotion_analysis.json` contains a successful emotion record, the header shows that emotion immediately after startup. If no successful emotion record exists, the header shows `情感状态：暂无`.

- [ ] **Step 6: Final status check**

Run:

```bash
git status --short
```

Expected: only intentional runtime data files may be modified or untracked. Do not commit `data/chat_history.json` or `data/emotion_analysis.json` unless the user explicitly asks.

---

## Self-Review

Spec coverage:

- Separate files for chat and emotion are preserved in Tasks 1 and 3.

- One startup response containing recent messages and latest emotion is implemented in Task 3.

- Frontend startup uses a single endpoint in Task 4.

- `ChatService.current_emotion` is initialized from persisted emotion in Tasks 2 and 3.

- Missing, damaged, empty, and unsuccessful emotion records are covered in Task 1.

- Full verification and local smoke testing are covered in Task 5.

Placeholder scan:

- The plan contains no placeholder markers or open-ended implementation steps.

Type consistency:

- `load_latest_successful_emotion()`, `initial_emotion`, `_session_snapshot()`, `GET /api/session`, `messages`, and `emotion` are named consistently across tasks.
