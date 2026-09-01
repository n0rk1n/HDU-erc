import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import chatbot.web as web
from chatbot.chat_service import ChatEvent
from chatbot.core.history import FeedbackUpdateResult, RegenerationUpdateResult
from chatbot.web import build_service, create_app, format_sse


class FakeRuntimeChatLlm:
    def invoke(self, prompt):
        return "{}"


class FakeService:
    def __init__(self):
        self.messages = []
        self.chain = "old-chain"
        self.chat_llm = FakeRuntimeChatLlm()
        self.config = type("Config", (), {"chat_llm": "config-chat-llm"})()

    def stream_reply(self, message):
        self.messages.append(message)
        yield ChatEvent("user_message", {"role": "human", "content": message})
        yield ChatEvent("token", {"content": "hi"})
        yield ChatEvent("done", {"content": "hi", "message_id": "ai_1"})

    def regenerate_reply(self, message_id, reason):
        return RegenerationUpdateResult(
            "updated",
            original_message_id=message_id,
            message_id="ai_regenerated",
            content="regenerated reply",
            reason=reason,
            original_user_message="hello",
        )


def test_build_service_does_not_duplicate_session_history(monkeypatch):
    from chatbot.core.llm import get_session_history, store

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
    monkeypatch.setattr(
        "chatbot.web.load_memory_config",
        lambda: type(
            "MemoryConfig",
            (),
            {"enabled": False, "db_path": "ignored", "max_results": 5},
        )(),
    )
    monkeypatch.setattr("chatbot.web.build_memory_provider", lambda config: object())
    store.clear()

    build_service()
    build_service()

    history = get_session_history("default")
    assert [message.content for message in history.messages] == ["hello", "hi"]


def test_build_service_passes_memory_provider(monkeypatch):
    captured = {}
    chat_llm = object()

    class FakeConfig:
        emotion_interval = 5
        chat_llm = "config-chat-llm"
        emotion_llm = object()

    class FakeService:
        def __init__(self, chain, config, emotion_llm, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("chatbot.web.load_config", lambda argv: FakeConfig())
    monkeypatch.setattr("chatbot.web.load_history", lambda: [])
    monkeypatch.setattr("chatbot.web.load_profile", lambda: {})
    monkeypatch.setattr("chatbot.web.format_profile", lambda profile: "")
    monkeypatch.setattr("chatbot.web.build_runtime_llms", lambda config: (chat_llm, object()))
    monkeypatch.setattr("chatbot.web.init_session_history", lambda session_id, records: None)
    monkeypatch.setattr("chatbot.web.build_chain", lambda llm, profile_text: object())
    monkeypatch.setattr("chatbot.web._latest_emotion_for_records", lambda records: None)
    monkeypatch.setattr("chatbot.web.load_memory_config", lambda: type(
        "MemoryConfig",
        (),
        {"enabled": False, "db_path": "ignored", "max_results": 3},
    )())
    monkeypatch.setattr("chatbot.web.build_memory_provider", lambda config: "memory-provider")
    monkeypatch.setattr("chatbot.web.ChatService", FakeService)

    service = web.build_service()

    assert service.chat_llm is chat_llm
    assert captured["memory_provider"] == "memory-provider"
    assert captured["memory_max_results"] == 3


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
        **kwargs,
    ):
        captured["initial_records"] = initial_records
        captured["initial_emotion"] = initial_emotion
        return SimpleNamespace()

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
    monkeypatch.setattr(
        "chatbot.web.load_memory_config",
        lambda: type(
            "MemoryConfig",
            (),
            {"enabled": False, "db_path": "ignored", "max_results": 5},
        )(),
    )
    monkeypatch.setattr("chatbot.web.build_memory_provider", lambda config: object())
    monkeypatch.setattr("chatbot.web.ChatService", fake_chat_service)

    build_service()

    assert captured["initial_records"] == records
    assert captured["initial_emotion"] == "sad"


def test_build_service_restores_latest_structured_emotion_state(monkeypatch):
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
        initial_emotion_state=None,
        session_id="default",
        **kwargs,
    ):
        captured["initial_emotion"] = initial_emotion
        captured["initial_emotion_state"] = initial_emotion_state
        return SimpleNamespace()

    monkeypatch.setattr("chatbot.web.load_config", lambda argv: object())
    monkeypatch.setattr("chatbot.web.load_history", lambda: records)
    monkeypatch.setattr(
        "chatbot.web.load_analysis_records",
        lambda: [{
            "timestamp": "t1",
            "turn_count": 5,
            "emotion_interval": 5,
            "input": "Dialogue context: q0</s>q1</s>q2</s>q3</s>q4",
            "emotion": "anxious",
            "success": True,
            "state": {
                "primary_emotion": "anxious",
                "confidence": 0.83,
                "secondary_emotions": ["apprehensive"],
                "evidence": "The user sounds worried.",
                "reply_strategy": "Use a calm tone.",
                "trajectory_note": "hopeful -> anxious",
                "safety_level": "normal",
            },
        }],
    )
    monkeypatch.setattr("chatbot.web.load_profile", lambda: {})
    monkeypatch.setattr("chatbot.web.format_profile", lambda profile: "")
    monkeypatch.setattr(
        "chatbot.web.build_runtime_llms",
        lambda config: (FakeLlm(), FakeLlm()),
    )
    monkeypatch.setattr("chatbot.web.build_chain", lambda llm, profile_text: object())
    monkeypatch.setattr(
        "chatbot.web.load_memory_config",
        lambda: type(
            "MemoryConfig",
            (),
            {"enabled": False, "db_path": "ignored", "max_results": 5},
        )(),
    )
    monkeypatch.setattr("chatbot.web.build_memory_provider", lambda config: object())
    monkeypatch.setattr("chatbot.web.ChatService", fake_chat_service)

    build_service()

    assert captured["initial_emotion"] == "anxious"
    assert captured["initial_emotion_state"].primary_emotion == "anxious"
    assert captured["initial_emotion_state"].confidence == 0.83
    assert captured["initial_emotion_state"].reply_strategy == "Use a calm tone."


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
        **kwargs,
    ):
        captured["initial_emotion"] = initial_emotion
        return SimpleNamespace()

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
    monkeypatch.setattr(
        "chatbot.web.load_memory_config",
        lambda: type(
            "MemoryConfig",
            (),
            {"enabled": False, "db_path": "ignored", "max_results": 5},
        )(),
    )
    monkeypatch.setattr("chatbot.web.build_memory_provider", lambda config: object())
    monkeypatch.setattr("chatbot.web.ChatService", fake_chat_service)

    build_service()

    assert captured["initial_emotion"] == ""


def test_profile_onboarding_questions_endpoint():
    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/profile/onboarding/questions")

    assert response.status_code == 200
    assert response.json() == {"questions": web.ONBOARDING_QUESTIONS}


def test_format_sse_encodes_event_and_json_data():
    output = format_sse(ChatEvent("token", {"content": "hi"}))

    assert output == 'event: token\ndata: {"content": "hi"}\n\n'


def test_emotion_feedback_request_payload_supports_pydantic_v1_dict():
    class LegacyRequest:
        def dict(self):
            return {"feedback": "accurate", "message_id": "ai_1"}

    assert web._request_payload(LegacyRequest()) == {
        "feedback": "accurate",
        "message_id": "ai_1",
    }


def test_legacy_stream_endpoint_is_removed():
    service = FakeService()
    app = create_app(service_factory=lambda: service)
    client = TestClient(app)

    response = client.get("/api/chat/stream?message=hello")

    assert response.status_code == 404
    assert service.messages == []


def test_static_assets_exist():
    root = Path(__file__).resolve().parents[2]

    assert (root / "chatbot" / "static" / "index.html").exists()
    assert (root / "chatbot" / "static" / "style.css").exists()
    assert (root / "chatbot" / "static" / "app.js").exists()


def test_static_app_js_loads_session_snapshot():
    root = Path(__file__).resolve().parents[2]
    app_js = (root / "chatbot" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/session?limit=10")' in app_js
    assert 'fetch("/api/history?limit=10")' not in app_js
    assert "payload.emotion" in app_js
    assert "情感状态：暂无" in app_js
    assert "renderEmotion(payload);" in app_js
    assert "emotionStatusEl.textContent = `情感状态：${payload.emotion}`;" not in app_js


def test_static_app_js_clears_safety_status_during_analysis_transitions():
    root = Path(__file__).resolve().parents[2]
    app_js = (root / "chatbot" / "static" / "app.js").read_text(encoding="utf-8")

    assert "function clearSafetyStatus()" in app_js
    assert app_js.count("clearSafetyStatus();") >= 5
    assert (
        'source.addEventListener("user_message", (event) => {\n'
        "    const payload = JSON.parse(event.data);\n"
        "    clearSafetyStatus();"
    ) in app_js
    assert (
        'source.addEventListener("emotion_start", () => {\n'
        '    emotionStatusEl.textContent = "情感状态：正在分析情绪…";\n'
        "    clearSafetyStatus();"
    ) in app_js
    assert (
        'source.addEventListener("emotion_error", () => {\n'
        '    emotionStatusEl.textContent = "情感状态：情感分析失败，本轮继续回复";\n'
        "    clearSafetyStatus();"
    ) in app_js


def test_static_app_js_initializes_from_session_snapshot():
    root = Path(__file__).resolve().parents[2]
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


def test_static_app_js_renders_and_submits_feedback_controls():
    root = Path(__file__).resolve().parents[2]
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

  insertBefore(child, nextSibling) {
    child.parent = this;
    const index = this.children.indexOf(nextSibling);
    if (index === -1) {
      this.children.push(child);
    } else {
      this.children.splice(index, 0, child);
    }
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
          emotion: {emotion: "sad"},
        }),
      };
    }
    if (url === "/api/messages/ai_1/feedback") {
      return {
        ok: true,
        json: async () => ({status: "updated", message_id: "ai_1", feedback: "like"}),
      };
    }
    if (url === "/api/emotion/feedback") {
      return {
        ok: true,
        json: async () => ({status: "saved"}),
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
    const dislikeButton = controls.children[1];
    const emotionButton = controls.children[3];
    if (likeButton.textContent !== "Good") {
      throw new Error(`unexpected like button text: ${likeButton.textContent}`);
    }
    if (likeButton.attributes["aria-label"] !== "Good") {
      throw new Error(`unexpected like aria-label: ${likeButton.attributes["aria-label"]}`);
    }
    if (dislikeButton.textContent !== "Bad") {
      throw new Error(`unexpected dislike button text: ${dislikeButton.textContent}`);
    }
    if (dislikeButton.attributes["aria-label"] !== "Bad") {
      throw new Error(`unexpected dislike aria-label: ${dislikeButton.attributes["aria-label"]}`);
    }
    if (emotionButton.textContent !== "Emotion?") {
      throw new Error(`unexpected emotion feedback text: ${emotionButton.textContent}`);
    }
    if (emotionButton.attributes["aria-label"] !== "Emotion correctness feedback") {
      throw new Error(`unexpected emotion aria-label: ${emotionButton.attributes["aria-label"]}`);
    }

    emotionButton.listeners.click();

    const emotionChoices = controls.children[4];
    const choiceLabels = emotionChoices.children.map((button) => button.textContent);
    if (JSON.stringify(choiceLabels) !== JSON.stringify(["Accurate", "Too positive", "Too negative", "Wrong"])) {
      throw new Error(`unexpected emotion choices: ${JSON.stringify(choiceLabels)}`);
    }

    await emotionChoices.children[2].listeners.click();

    if (fetchCalls[1].url !== "/api/emotion/feedback") {
      throw new Error(`unexpected emotion feedback url: ${fetchCalls[1].url}`);
    }
    if (fetchCalls[1].options.method !== "POST") {
      throw new Error(`unexpected emotion feedback method: ${fetchCalls[1].options.method}`);
    }
    if (fetchCalls[1].options.body !== JSON.stringify({
      message_id: "ai_1",
      feedback: "too_negative",
      predicted_emotion: "sad",
    })) {
      throw new Error(`unexpected emotion feedback body: ${fetchCalls[1].options.body}`);
    }

    await likeButton.listeners.click();

    if (fetchCalls[2].url !== "/api/messages/ai_1/feedback") {
      throw new Error(`unexpected feedback url: ${fetchCalls[2].url}`);
    }
    if (fetchCalls[2].options.method !== "POST") {
      throw new Error(`unexpected feedback method: ${fetchCalls[2].options.method}`);
    }
    if (fetchCalls[2].options.body !== JSON.stringify({feedback: "like"})) {
      throw new Error(`unexpected feedback body: ${fetchCalls[2].options.body}`);
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


def test_static_app_js_regenerates_reply_with_reason_and_collapses_original():
    root = Path(__file__).resolve().parents[2]
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

  insertBefore(child, nextSibling) {
    child.parent = this;
    const index = this.children.indexOf(nextSibling);
    if (index === -1) {
      this.children.push(child);
    } else {
      this.children.splice(index, 0, child);
    }
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

  get nextSibling() {
    if (!this.parent) {
      return null;
    }
    const index = this.parent.children.indexOf(this);
    return this.parent.children[index + 1] || null;
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
let resolveRegenerate;
const regenerateResponse = new Promise((resolve) => {
  resolveRegenerate = resolve;
});

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
            {role: "human", content: "q1"},
            {role: "ai", content: "bad", id: "ai_1", feedback: null},
          ],
          emotion: null,
        }),
      };
    }
    if (url === "/api/messages/ai_1/regenerate") {
      return regenerateResponse;
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
    const original = messagesEl.children[1];
    const controls = original.children[1];
    const regenerateButton = controls.children[2];
    if (regenerateButton.textContent !== "Regenerate") {
      throw new Error(`unexpected regenerate text: ${regenerateButton.textContent}`);
    }

    regenerateButton.listeners.click();
    const reasons = controls.children[4];
    const firstReason = reasons.children[0];
    const secondReason = reasons.children[1];
    if (firstReason.textContent !== "不准确") {
      throw new Error(`unexpected first reason: ${firstReason.textContent}`);
    }

    const pendingRegeneration = firstReason.listeners.click();
    if (!firstReason.disabled) {
      throw new Error("selected reason should be disabled while regenerate is pending");
    }
    if (!secondReason.disabled) {
      throw new Error("other reasons should be disabled while regenerate is pending");
    }
    resolveRegenerate({
      ok: true,
      json: async () => ({
        status: "regenerated",
        original_message_id: "ai_1",
        message_id: "ai_2",
        content: "better",
        reason: "不准确",
      }),
    });
    await pendingRegeneration;

    if (!original.className.includes("regenerated")) {
      throw new Error(`original should be collapsed: ${original.className}`);
    }
    if (messagesEl.children.length !== 3) {
      throw new Error(`expected regenerated message inserted, got ${messagesEl.children.length}`);
    }
    if (messagesEl.children[2].children[0].textContent !== "better") {
      throw new Error("regenerated message content missing");
    }
    if (fetchCalls[1].url !== "/api/messages/ai_1/regenerate") {
      throw new Error(`unexpected regenerate url: ${fetchCalls[1].url}`);
    }
    if (fetchCalls[1].options.body !== JSON.stringify({reason: "不准确"})) {
      throw new Error(`unexpected regenerate body: ${fetchCalls[1].options.body}`);
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


def test_static_app_js_recovers_controls_when_regeneration_fails():
    root = Path(__file__).resolve().parents[2]
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

  insertBefore(child, nextSibling) {
    child.parent = this;
    const index = this.children.indexOf(nextSibling);
    if (index === -1) {
      this.children.push(child);
    } else {
      this.children.splice(index, 0, child);
    }
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

  get nextSibling() {
    if (!this.parent) {
      return null;
    }
    const index = this.parent.children.indexOf(this);
    return this.parent.children[index + 1] || null;
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
let resolveRegenerate;
const regenerateResponse = new Promise((resolve) => {
  resolveRegenerate = resolve;
});

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
            {role: "human", content: "q1"},
            {role: "ai", content: "bad", id: "ai_1", feedback: null},
          ],
          emotion: null,
        }),
      };
    }
    if (url === "/api/messages/ai_1/regenerate") {
      return regenerateResponse;
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
    const original = messagesEl.children[1];
    const controls = original.children[1];
    const likeButton = controls.children[0];
    const dislikeButton = controls.children[1];
    const regenerateButton = controls.children[2];

    regenerateButton.listeners.click();
    const reasons = controls.children[4];
    const firstReason = reasons.children[0];
    const secondReason = reasons.children[1];

    const pendingRegeneration = firstReason.listeners.click();
    const disabledButtons = [
      likeButton,
      dislikeButton,
      regenerateButton,
      firstReason,
      secondReason,
    ].filter((button) => button.disabled);
    if (disabledButtons.length !== 5) {
      throw new Error(`expected controls disabled while pending, got ${disabledButtons.length}`);
    }

    resolveRegenerate({ok: false, json: async () => ({})});
    await pendingRegeneration;

    const status = controls.children[5];
    if (status.textContent !== "重新生成失败") {
      throw new Error(`unexpected failure status: ${status.textContent}`);
    }
    const reenabledButtons = [
      likeButton,
      dislikeButton,
      regenerateButton,
      firstReason,
      secondReason,
    ].filter((button) => !button.disabled);
    if (reenabledButtons.length !== 5) {
      throw new Error(`expected controls re-enabled after failure, got ${reenabledButtons.length}`);
    }
    if (original.className.includes("regenerated")) {
      throw new Error(`original should not be collapsed after failure: ${original.className}`);
    }
    if (messagesEl.children.length !== 2) {
      throw new Error(`expected no regenerated message inserted, got ${messagesEl.children.length}`);
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


def test_static_app_js_disables_visible_reasons_during_pending_feedback():
    root = Path(__file__).resolve().parents[2]
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

  insertBefore(child, nextSibling) {
    child.parent = this;
    const index = this.children.indexOf(nextSibling);
    if (index === -1) {
      this.children.push(child);
    } else {
      this.children.splice(index, 0, child);
    }
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

  get nextSibling() {
    if (!this.parent) {
      return null;
    }
    const index = this.parent.children.indexOf(this);
    return this.parent.children[index + 1] || null;
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
let resolveFeedback;
const feedbackResponse = new Promise((resolve) => {
  resolveFeedback = resolve;
});

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
            {role: "human", content: "q1"},
            {role: "ai", content: "bad", id: "ai_1", feedback: null},
          ],
          emotion: null,
        }),
      };
    }
    if (url === "/api/messages/ai_1/feedback") {
      return feedbackResponse;
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
    const original = messagesEl.children[1];
    const controls = original.children[1];
    const likeButton = controls.children[0];
    const dislikeButton = controls.children[1];
    const regenerateButton = controls.children[2];

    regenerateButton.listeners.click();
    const reasons = controls.children[4];
    const firstReason = reasons.children[0];
    const secondReason = reasons.children[1];

    const pendingFeedback = likeButton.listeners.click();
    const disabledButtons = [
      likeButton,
      dislikeButton,
      regenerateButton,
      firstReason,
      secondReason,
    ].filter((button) => button.disabled);
    if (disabledButtons.length !== 5) {
      throw new Error(`expected visible controls disabled during feedback, got ${disabledButtons.length}`);
    }

    resolveFeedback({ok: false, json: async () => ({})});
    await pendingFeedback;

    const status = controls.children[5];
    if (status.textContent !== "评价保存失败") {
      throw new Error(`unexpected feedback failure status: ${status.textContent}`);
    }
    const reenabledButtons = [
      likeButton,
      dislikeButton,
      regenerateButton,
      firstReason,
      secondReason,
    ].filter((button) => !button.disabled);
    if (reenabledButtons.length !== 5) {
      throw new Error(`expected visible controls re-enabled after feedback failure, got ${reenabledButtons.length}`);
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


def test_static_app_js_renders_regenerated_session_reply_after_original():
    root = Path(__file__).resolve().parents[2]
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

  insertBefore(child, nextSibling) {
    child.parent = this;
    const index = this.children.indexOf(nextSibling);
    if (index === -1) {
      this.children.push(child);
    } else {
      this.children.splice(index, 0, child);
    }
    return child;
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
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
    if (url === "/api/session?limit=10") {
      return {
        ok: true,
        json: async () => ({
          messages: [
            {role: "human", content: "q1"},
            {
              role: "ai",
              content: "bad",
              id: "ai_1",
              feedback: null,
              regeneration: {message_id: "ai_2", reason: "不准确"},
            },
            {role: "human", content: "q2"},
            {role: "ai", content: "other", id: "ai_3", feedback: null},
            {role: "ai", content: "better", id: "ai_2", feedback: null, regenerated_from: "ai_1"},
          ],
          emotion: null,
        }),
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

setImmediate(() => {
  try {
    const contents = messagesEl.children.map((message) => message.children[0].textContent);
    const expected = ["q1", "bad", "better", "q2", "other"];
    if (JSON.stringify(contents) !== JSON.stringify(expected)) {
      throw new Error(`unexpected rendered order: ${JSON.stringify(contents)}`);
    }
    const original = messagesEl.children[1];
    const regenerated = messagesEl.children[2];
    if (!original.className.includes("regenerated")) {
      throw new Error(`original should be collapsed: ${original.className}`);
    }
    if (regenerated.className.includes("regenerated")) {
      throw new Error(`regenerated reply should render normally: ${regenerated.className}`);
    }
    if (regenerated.children[0].textContent !== "better") {
      throw new Error(`unexpected regenerated content: ${regenerated.children[0].textContent}`);
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


def test_superseded_global_routes_are_removed():
    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    requests = [
        ("get", "/api/history", None),
        ("get", "/api/session", None),
        ("get", "/api/profile", None),
        ("put", "/api/profile", {"profile": {}}),
        ("post", "/api/profile/onboarding/draft", {"answers": []}),
        ("get", "/api/emotion/timeline", None),
        ("post", "/api/messages/ai_1/feedback", {"feedback": "like"}),
        ("post", "/api/emotion/feedback", {"feedback": "accurate"}),
        ("post", "/api/messages/ai_1/regenerate", {"reason": "其他"}),
    ]

    for method, path, payload in requests:
        response = getattr(client, method)(path, json=payload) if payload is not None else getattr(client, method)(path)
        assert response.status_code == 404, path
