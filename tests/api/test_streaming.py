from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage

from chatbot.db.messages import MessageRepository
from chatbot.llm.types import ModelDelta
from chatbot.services.events import SseEvent, TurnSubscription
from tests.emotion.helpers import create_app
from chatbot.web import format_sse, stream_subscription
from tests.api.helpers import OfflineModel, parse_sse


def test_post_stream_emits_fetch_compatible_ordered_unicode_sse(
    client: TestClient, resolved_user, offline_model
) -> None:
    """Catches malformed framing, reordered events, or escaped Unicode token data."""
    user_id = resolved_user["user"]["id"]
    request_id = str(uuid4())

    response = client.post(
        f"/api/users/{user_id}/messages:stream",
        json={"request_id": request_id, "content": "你好"},
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "charset=utf-8" in response.headers["content-type"].lower()
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    frames = parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "run_started",
        "user_message",
        "progress",
        "progress",
        "progress",
        "token",
        "token",
        "done",
    ]
    assert frames[0][1]["request_id"] == request_id
    assert frames[1][1]["content"] == "你好"
    assert [data["content"] for name, data in frames if name == "token"] == ["你", "好"]
    assert frames[-1][1]["message"]["status"] == "completed"
    assert "\\u4f60" not in response.text
    assert response.text.endswith("\n\n")
    assert offline_model.calls == 1
    for forbidden in (
        "reasoning",
        "prompt",
        "parameters",
        "trace",
        "thread_id",
        "api_key",
        "authorization",
        "sk-secret",
    ):
        assert forbidden not in response.text.lower()


def test_stream_errors_before_headers_are_safe_http_json(
    client: TestClient, resolved_user
) -> None:
    """Catches domain and request-validation errors being converted into 200 SSE streams."""
    valid = {"request_id": str(uuid4()), "content": "hello"}
    missing_user = client.post("/api/users/999999/messages:stream", json=valid)
    assert missing_user.status_code == 404
    assert missing_user.headers["content-type"].startswith("application/json")
    assert missing_user.json() == {
        "error": {"code": "user_not_found", "message": "user not found"}
    }

    invalid_payloads = (
        {"request_id": "not-a-uuid", "content": "hello"},
        {"request_id": "00000000-0000-1000-8000-000000000001", "content": "hello"},
        {"request_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", "content": "hello"},
        {"request_id": str(uuid4()), "content": " \n "},
        {"request_id": str(uuid4()), "content": "bad\x00value"},
        {"request_id": str(uuid4()), "content": 7},
    )
    user_id = resolved_user["user"]["id"]
    for payload in invalid_payloads:
        response = client.post(
            f"/api/users/{user_id}/messages:stream", json=payload
        )
        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {
            "error": {
                "code": "invalid_identifier_or_message",
                "message": "invalid identifier or message",
            }
        }


def test_model_failure_after_headers_is_only_a_safe_error_event(
    app_config,
) -> None:
    """Catches an in-stream failure changing HTTP status or leaking the raw exception."""
    from tests.emotion.helpers import create_app

    app = create_app(config=app_config, model=OfflineModel(fail=True))
    with TestClient(app) as client:
        user = client.post(
            "/api/users/resolve", json={"identifier": "failure"}
        ).json()["user"]
        response = client.post(
            f"/api/users/{user['id']}/messages:stream",
            json={"request_id": str(uuid4()), "content": "hello"},
        )

    assert response.status_code == 200
    frames = parse_sse(response.text)
    assert [name for name, _ in frames] == [
        "run_started",
        "user_message",
        "progress",
        "progress",
        "progress",
        "token",
        "error",
    ]
    assert frames[-1] == (
        "error",
        {"code": "model_error", "message": "model generation failed"},
    )
    assert "raw model failure" not in response.text
    assert "sk-secret" not in response.text


@pytest.mark.asyncio
async def test_stream_adapter_turns_iterator_failure_into_error_event_and_detaches() -> None:
    """Catches post-header iterator exceptions escaping ASGI or retaining subscribers."""
    class BrokenSubscription:
        def __init__(self) -> None:
            self.detached = False

        async def events(self):
            yield SseEvent("token", {"content": "一"})
            raise RuntimeError("SQL secret traceback")

        def detach(self) -> None:
            self.detached = True

    subscription = BrokenSubscription()
    chunks = [chunk async for chunk in stream_subscription(subscription)]

    assert chunks[0] == format_sse(SseEvent("token", {"content": "一"}))
    assert parse_sse("".join(chunks))[-1] == (
        "error",
        {"code": "database_error", "message": "database error"},
    )
    assert subscription.detached is True
    assert "SQL secret traceback" not in "".join(chunks)


@pytest.mark.asyncio
async def test_cancelled_stream_detaches_without_cancelling_producer() -> None:
    """Catches client cancellation being swallowed or forwarded into the producer."""
    producer_cancelled = False

    class BlockingSubscription:
        def __init__(self) -> None:
            self.detached = False

        async def events(self):
            nonlocal producer_cancelled
            yield SseEvent("run_started", {"request_id": "safe"})
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                producer_cancelled = True
                raise

        def detach(self) -> None:
            self.detached = True

    subscription = BlockingSubscription()
    stream = stream_subscription(subscription)
    assert json.loads((await anext(stream)).split("data: ", 1)[1]) == {
        "request_id": "safe"
    }
    await stream.aclose()

    assert subscription.detached is True
    assert producer_cancelled is False


def test_asgi_disconnect_detaches_http_stream_but_background_turn_completes(
    app_config, monkeypatch
) -> None:
    """Catches Starlette disconnect cancellation propagating into the turn producer."""
    class SlowModel:
        provider = "asgi-test"
        parameters = {"model": "asgi-test"}

        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = False

        async def stream(
            self, prompt: Sequence[BaseMessage]
        ) -> AsyncIterator[ModelDelta]:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            yield ModelDelta(content="后台完成")

    detached: list[TurnSubscription] = []
    original_detach = TurnSubscription.detach

    def record_detach(subscription: TurnSubscription) -> None:
        detached.append(subscription)
        original_detach(subscription)

    monkeypatch.setattr(TurnSubscription, "detach", record_detach)
    model = SlowModel()
    app = create_app(config=app_config, model=model)
    request_id = str(uuid4())
    with TestClient(app) as client:
        user_id = client.post(
            "/api/users/resolve", json={"identifier": "asgi-disconnect"}
        ).json()["user"]["id"]

        async def exercise_disconnect():
            body = json.dumps(
                {"request_id": request_id, "content": "keep working"}
            ).encode()
            receives: asyncio.Queue[dict[str, object]] = asyncio.Queue()
            await receives.put(
                {"type": "http.request", "body": body, "more_body": False}
            )
            sent: list[dict[str, object]] = []
            disconnect_sent = False

            async def receive() -> dict[str, object]:
                return await receives.get()

            async def send(message: dict[str, object]) -> None:
                nonlocal disconnect_sent
                sent.append(message)
                if (
                    message["type"] == "http.response.body"
                    and message.get("body")
                    and not disconnect_sent
                ):
                    disconnect_sent = True
                    await receives.put({"type": "http.disconnect"})

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "scheme": "http",
                "method": "POST",
                "root_path": "",
                "path": f"/api/users/{user_id}/messages:stream",
                "raw_path": f"/api/users/{user_id}/messages:stream".encode(),
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"accept", b"text/event-stream"),
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            }
            response_task = asyncio.create_task(app(scope, receive, send))
            await asyncio.wait_for(model.entered.wait(), timeout=1)
            await asyncio.wait_for(response_task, timeout=1)

            assert model.cancelled is False
            assert model.release.is_set() is False
            assert detached
            assert any(message["type"] == "http.response.start" for message in sent)
            assert not any(
                message["type"] == "http.response.body"
                and message.get("more_body") is False
                for message in sent
            )

            model.release.set()
            await app.state.coordinator.shutdown(timeout_seconds=1)
            conversation = await app.state.conversations.get_default_by_user(user_id)
            assert conversation is not None
            assistant = await MessageRepository(app.state.database).find_assistant(
                conversation.id,
                request_id,
            )
            return assistant

        assistant = client.portal.call(exercise_disconnect)

    assert model.cancelled is False
    assert assistant is not None
    assert assistant.status == "completed"
    assert assistant.content == "后台完成"
