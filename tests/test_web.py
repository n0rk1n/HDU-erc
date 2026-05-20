import json
from pathlib import Path

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


def test_session_endpoint_returns_null_emotion(monkeypatch):
    monkeypatch.setattr("chatbot.web.load_history", lambda: [])
    monkeypatch.setattr("chatbot.web.load_latest_successful_emotion", lambda: None)

    app = create_app(service_factory=lambda: FakeService())
    client = TestClient(app)

    response = client.get("/api/session?limit=10")

    assert response.status_code == 200
    assert response.json() == {"messages": [], "emotion": None}


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
