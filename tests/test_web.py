import json

from fastapi.testclient import TestClient

from chatbot.chat_service import ChatEvent
from chatbot.web import create_app, format_sse


class FakeService:
    def __init__(self):
        self.messages = []

    def stream_reply(self, message):
        self.messages.append(message)
        yield ChatEvent("user_message", {"role": "human", "content": message})
        yield ChatEvent("token", {"content": "hi"})
        yield ChatEvent("done", {"content": "hi"})


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
