import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatbot.chat_service import ChatEvent
from chatbot.web import build_service, create_app, format_sse


class FakeService:
    def __init__(self):
        self.messages = []

    def stream_reply(self, message):
        self.messages.append(message)
        yield ChatEvent("user_message", {"role": "human", "content": message})
        yield ChatEvent("token", {"content": "hi"})
        yield ChatEvent("done", {"content": "hi"})


def test_build_service_does_not_duplicate_session_history(monkeypatch):
    from chatbot.llm import get_session_history, store

    records = [
        {"role": "human", "content": "hello"},
        {"role": "ai", "content": "hi"},
    ]

    class FakeLlm:
        pass

    monkeypatch.setattr("chatbot.web.load_config", lambda argv: object())
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr("chatbot.web.load_profile", lambda: {})
    monkeypatch.setattr("chatbot.web.format_profile", lambda profile: "")
    monkeypatch.setattr(
        "chatbot.web.build_runtime_llms",
        lambda config: (FakeLlm(), FakeLlm()),
    )
    monkeypatch.setattr("chatbot.web.build_chain", lambda llm, profile_text: object())
    store.clear()

    build_service()
    build_service()

    history = get_session_history("default")
    assert [message.content for message in history.messages] == ["hello", "hi"]


def test_build_service_uses_latest_successful_emotion(monkeypatch):
    records = [
        {"role": "human", "content": f"q{i}"}
        for i in range(5)
    ]

    class FakeLlm:
        pass

    captured = {}

    def fake_chat_service(
        chain,
        config,
        emotion_llm,
        initial_records=None,
        initial_emotion="",
        session_id="default",
    ):
        captured["initial_records"] = initial_records
        captured["initial_emotion"] = initial_emotion
        return object()

    monkeypatch.setattr("chatbot.web.load_config", lambda argv: object())
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "t1",
            "turn_count": 5,
            "emotion_interval": 5,
            "input": "Dialogue context: q0</s>q1</s>q2</s>q3</s>q4",
            "emotion": "sad",
            "success": True,
        }],
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


def test_build_service_ignores_emotion_when_history_is_too_short(monkeypatch):
    records = [
        {"role": "human", "content": "hello"},
        {"role": "ai", "content": "hi"},
    ]

    class FakeLlm:
        pass

    captured = {}

    def fake_chat_service(
        chain,
        config,
        emotion_llm,
        initial_records=None,
        initial_emotion="",
        session_id="default",
    ):
        captured["initial_emotion"] = initial_emotion
        return object()

    monkeypatch.setattr("chatbot.web.load_config", lambda argv: object())
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "t1",
            "turn_count": 5,
            "emotion_interval": 5,
            "input": "Dialogue context: q0</s>q1</s>q2</s>q3</s>q4",
            "emotion": "sad",
            "success": True,
        }],
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

    assert captured["initial_emotion"] == ""


def test_session_endpoint_returns_messages_and_latest_emotion(monkeypatch):
    records = [
        {"role": "human", "content": f"q{i}", "timestamp": f"t{i}"}
        for i in range(12)
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "emotion-time",
            "turn_count": 5,
            "emotion_interval": 5,
            "input": "Dialogue context: q0</s>q1</s>q2</s>q3</s>q4",
            "emotion": "sad",
            "success": True,
        }],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["messages"][0] == {
        "role": "human",
        "content": "q2",
        "timestamp": "t2",
    }
    assert payload["messages"][-1] == {
        "role": "human",
        "content": "q11",
        "timestamp": "t11",
    }
    assert payload["emotion"] == {
        "emotion": "sad",
        "timestamp": "emotion-time",
        "turn_count": 5,
    }


def test_session_endpoint_matches_emotion_after_restart_turn_count(monkeypatch):
    records = [
        {"role": "human", "content": "old q1", "timestamp": "t1"},
        {"role": "ai", "content": "old a1", "timestamp": "t2"},
        {"role": "human", "content": "old q2", "timestamp": "t3"},
        {"role": "ai", "content": "old a2", "timestamp": "t4"},
        {"role": "human", "content": "new q1", "timestamp": "t5"},
        {"role": "ai", "content": "new a1", "timestamp": "t6"},
        {"role": "human", "content": "new q2", "timestamp": "t7"},
        {"role": "ai", "content": "new a2", "timestamp": "t8"},
        {"role": "human", "content": "new q3", "timestamp": "t9"},
        {"role": "ai", "content": "new a3", "timestamp": "t10"},
        {"role": "human", "content": "new q4", "timestamp": "t11"},
        {"role": "ai", "content": "new a4", "timestamp": "t12"},
        {"role": "human", "content": "new q5", "timestamp": "t13"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "emotion-time",
            "turn_count": 5,
            "emotion_interval": 5,
            "input": (
                "Dialogue context: old q2</s>old a2</s>new q1</s>new a1</s>"
                "new q2</s>new a2</s>new q3</s>new a3</s>new q4</s>new a4</s>new q5"
            ),
            "emotion": "sad",
            "success": True,
        }],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json()["emotion"] == {
        "emotion": "sad",
        "timestamp": "emotion-time",
        "turn_count": 5,
    }


def test_session_endpoint_does_not_fall_back_when_latest_emotion_mismatches(monkeypatch):
    records = [
        {"role": "human", "content": "q1", "timestamp": "t1"},
        {"role": "ai", "content": "a1", "timestamp": "t2"},
        {"role": "human", "content": "q2", "timestamp": "t3"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [
            {
                "timestamp": "older-emotion-time",
                "turn_count": 1,
                "emotion_interval": 1,
                "input": "Dialogue context: q1",
                "emotion": "anxious",
                "success": True,
            },
            {
                "timestamp": "latest-emotion-time",
                "turn_count": 1,
                "emotion_interval": 1,
                "input": "Dialogue context: different q",
                "emotion": "sad",
                "success": True,
            },
        ],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json()["emotion"] is None


def test_session_endpoint_returns_null_emotion(monkeypatch):
    monkeypatch.setattr("chatbot.web.load_history", lambda: [])
    monkeypatch.setattr("chatbot.web.load_analysis_records", lambda: [])

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json() == {"messages": [], "emotion": None}


def test_session_endpoint_returns_null_for_stale_emotion(monkeypatch):
    records = [
        {"role": "human", "content": "q1", "timestamp": "t1"},
        {"role": "ai", "content": "a1", "timestamp": "t2"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "emotion-time",
            "turn_count": 5,
            "emotion_interval": 5,
            "input": "Dialogue context: q1",
            "emotion": "sad",
            "success": True,
        }],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [
            {"role": "human", "content": "q1", "timestamp": "t1"},
            {"role": "ai", "content": "a1", "timestamp": "t2"},
        ],
        "emotion": None,
    }


def test_session_endpoint_returns_null_for_replaced_history(monkeypatch):
    records = [
        {"role": "human", "content": "new q1", "timestamp": "t1"},
        {"role": "ai", "content": "new a1", "timestamp": "t2"},
        {"role": "human", "content": "new q2", "timestamp": "t3"},
        {"role": "ai", "content": "new a2", "timestamp": "t4"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "emotion-time",
            "turn_count": 2,
            "emotion_interval": 2,
            "input": "Dialogue context: old q1</s>old a1</s>old q2",
            "emotion": "sad",
            "success": True,
        }],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [
            {"role": "human", "content": "new q1", "timestamp": "t1"},
            {"role": "ai", "content": "new a1", "timestamp": "t2"},
            {"role": "human", "content": "new q2", "timestamp": "t3"},
            {"role": "ai", "content": "new a2", "timestamp": "t4"},
        ],
        "emotion": None,
    }


def test_session_endpoint_returns_null_for_substring_collision(monkeypatch):
    records = [
        {"role": "human", "content": "q1", "timestamp": "t1"},
        {"role": "ai", "content": "a1", "timestamp": "t2"},
        {"role": "human", "content": "q2", "timestamp": "t3"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "emotion-time",
            "turn_count": 2,
            "emotion_interval": 2,
            "input": "Dialogue context: q10</s>a10</s>q20",
            "emotion": "sad",
            "success": True,
        }],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json()["emotion"] is None


def test_session_endpoint_requires_exact_dialogue_context(monkeypatch):
    records = [
        {"role": "human", "content": "q1", "timestamp": "t1"},
        {"role": "ai", "content": "a1", "timestamp": "t2"},
        {"role": "human", "content": "q2", "timestamp": "t3"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "emotion-time",
            "turn_count": 2,
            "emotion_interval": 2,
            "input": "Dialogue context: q1</s>a1</s>q2 extra",
            "emotion": "sad",
            "success": True,
        }],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json()["emotion"] is None


def test_session_endpoint_checks_full_emotion_prompt_window(monkeypatch):
    records = [
        {"role": "human", "content": "q1", "timestamp": "t1"},
        {"role": "ai", "content": "a1", "timestamp": "t2"},
        {"role": "human", "content": "new q2", "timestamp": "t3"},
        {"role": "ai", "content": "a2", "timestamp": "t4"},
        {"role": "human", "content": "q3", "timestamp": "t5"},
        {"role": "ai", "content": "a3", "timestamp": "t6"},
        {"role": "human", "content": "q4", "timestamp": "t7"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "emotion-time",
            "turn_count": 4,
            "emotion_interval": 2,
            "input": "Dialogue context: old q2</s>a2</s>q3</s>a3</s>q4",
            "emotion": "sad",
            "success": True,
        }],
    )

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json()["emotion"] is None


def test_format_sse_encodes_event_and_json_data():
    output = format_sse(ChatEvent("token", {"content": "hi"}))

    assert output == 'event: token\ndata: {"content": "hi"}\n\n'


def test_history_endpoint_returns_recent_structured_messages(monkeypatch):
    records = [
        {"role": "human", "content": f"q{i}", "timestamp": f"t{i}"}
        for i in range(12)
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/history?limit=10")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [
            {"role": "human", "content": "q2", "timestamp": "t2"},
            {"role": "human", "content": "q3", "timestamp": "t3"},
            {"role": "human", "content": "q4", "timestamp": "t4"},
            {"role": "human", "content": "q5", "timestamp": "t5"},
            {"role": "human", "content": "q6", "timestamp": "t6"},
            {"role": "human", "content": "q7", "timestamp": "t7"},
            {"role": "human", "content": "q8", "timestamp": "t8"},
            {"role": "human", "content": "q9", "timestamp": "t9"},
            {"role": "human", "content": "q10", "timestamp": "t10"},
            {"role": "human", "content": "q11", "timestamp": "t11"},
        ]
    }


def test_history_endpoint_filters_roles_before_limiting(monkeypatch):
    records = [
        {"role": "human", "content": f"q{i}", "timestamp": f"t{i}"}
        for i in range(10)
    ] + [
        {"role": "system", "content": "ignored", "timestamp": "ignored"},
    ]
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/history?limit=10")

    assert response.status_code == 200
    assert len(response.json()["messages"]) == 10
    assert response.json()["messages"][0]["content"] == "q0"
    assert response.json()["messages"][-1]["content"] == "q9"


def test_stream_endpoint_returns_sse_events():
    service = FakeService()
    app = create_app(service_factory=lambda: service)
    client = TestClient(app)

    response = client.get("/api/chat/stream?message=hello")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: user_message" in response.text
    assert 'data: {"role": "human", "content": "hello"}' in response.text
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert service.messages == ["hello"]


def test_stream_endpoint_rejects_blank_message():
    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/chat/stream?message=%20%20")

    assert response.status_code == 400
    assert response.json() == {"detail": "Message must not be empty."}


def test_static_assets_exist():
    root = Path(__file__).resolve().parents[1]

    assert (root / "chatbot" / "static" / "index.html").exists()
    assert (root / "chatbot" / "static" / "style.css").exists()
    assert (root / "chatbot" / "static" / "app.js").exists()


def test_static_app_js_loads_session_snapshot():
    root = Path(__file__).resolve().parents[1]
    app_js = (root / "chatbot" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/session?limit=10")' in app_js
    assert 'fetch("/api/history?limit=10")' not in app_js
    assert "payload.emotion" in app_js
    assert "情感状态：暂无" in app_js
    assert "renderEmotion(payload);" in app_js
    assert "emotionStatusEl.textContent = `情感状态：${payload.emotion}`;" not in app_js


def test_static_app_js_initializes_from_session_snapshot():
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
    this.textContent = "";
    this.className = "";
    this.disabled = false;
    this.value = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.listeners = {};
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
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
  fetch: async (url) => {
    fetchCalls.push(url);
    return {
      ok: true,
      json: async () => ({
        messages: [
          {role: "human", content: "hello"},
          {role: "ai", content: "hi"},
        ],
        emotion: {emotion: "sad"},
      }),
    };
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

setImmediate(() => {
  try {
    if (fetchCalls.length !== 1 || fetchCalls[0] !== "/api/session?limit=10") {
      throw new Error(`unexpected fetch calls: ${JSON.stringify(fetchCalls)}`);
    }
    if (messagesEl.children.length !== 2) {
      throw new Error(`expected 2 rendered messages, got ${messagesEl.children.length}`);
    }
    const contents = messagesEl.children.map((message) => message.children[0].textContent);
    if (JSON.stringify(contents) !== JSON.stringify(["hello", "hi"])) {
      throw new Error(`unexpected rendered messages: ${JSON.stringify(contents)}`);
    }
    if (emotionStatusEl.textContent !== "情感状态：sad") {
      throw new Error(`unexpected emotion status: ${emotionStatusEl.textContent}`);
    }
    if (inputEl.disabled !== false || buttonEl.disabled !== false) {
      throw new Error("input and button should be unlocked after initialization");
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


def test_index_endpoint_returns_html():
    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_static_app_js_is_served():
    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
