from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import get_args, get_type_hints
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

import chatbot.web as web
import chatbot.models.graph as graph_models
from chatbot.models import GraphEvent
from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.graphs.runtime import build_graph_runtime
from chatbot.web import adapt_graph_stream


EMOTION_JSON = (
    '{"primary_emotion":"content","confidence":0.8,'
    '"secondary_emotions":[],"evidence":"steady",'
    '"reply_strategy":"respond naturally","trajectory_note":"stable",'
    '"safety_level":"normal"}'
)
TURN_REQUEST_ID = "00000000-0000-4000-8000-000000000001"
REGEN_REQUEST_ID = "00000000-0000-4000-8000-000000000002"


class SequenceModel:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, value, config=None, **kwargs):
        index = min(self.calls, len(self.responses) - 1)
        response = self.responses[index]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)


@contextmanager
def _client(tmp_path, monkeypatch, chat_model: SequenceModel):
    llm = LlmConfig(provider="test", api_key="test", model="test", temperature=0.0)
    chat_config = ChatConfig(chat_llm=llm, emotion_llm=llm, emotion_interval=5)
    graph_config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="test-signing-secret-that-is-at-least-32-bytes",
    )
    emotion_model = SequenceModel(EMOTION_JSON)
    monkeypatch.setattr(web, "load_config", lambda argv: chat_config)
    monkeypatch.setattr(web, "load_graph_config", lambda: graph_config)
    monkeypatch.setattr(
        web,
        "build_graph_runtime",
        lambda handles, chat, graph: build_graph_runtime(
            handles,
            chat,
            graph,
            model_factory=lambda config: (chat_model, emotion_model),
            now=lambda: datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
        ),
    )
    with TestClient(web.create_app(service_factory=lambda: object())) as client:
        yield client, emotion_model


def _bootstrap(client: TestClient) -> tuple[str, str]:
    response = client.post("/api/clients/bootstrap")
    assert response.status_code == 201
    payload = response.json()
    return payload["client_id"], payload["thread"]["thread_id"]


def _events(response_text: str) -> list[dict]:
    events = []
    for block in response_text.split("\n\n"):
        lines = block.splitlines()
        if len(lines) != 2 or not lines[0].startswith("event: "):
            continue
        events.append(
            {
                "event": lines[0].removeprefix("event: "),
                "data": json.loads(lines[1].removeprefix("data: ")),
            }
        )
    return events


def test_request_result_declares_durable_done_event_data():
    assert hasattr(graph_models, "CompletionEventData")
    assert (
        graph_models.RequestResult.__annotations__["event_data"]
        is graph_models.CompletionEventData
    )


def test_graph_event_type_allows_exactly_the_public_event_names():
    assert set(get_args(get_type_hints(GraphEvent)["event"])) == {
        "run_started",
        "user_message",
        "emotion_start",
        "emotion_done",
        "emotion_error",
        "safety",
        "token",
        "done",
        "error",
    }


async def _chunks(*items: dict) -> AsyncIterator[dict]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_event_adapter_filters_non_generation_message_chunks():
    chunks = _chunks(
        {"type": "custom", "data": {"event": "emotion_start", "data": {}}},
        {
            "type": "custom",
            "data": {
                "event": "emotion_done",
                "data": {
                    "emotion": "anxious",
                    "state": {
                        "primary_emotion": "anxious",
                        "safety_level": "supportive",
                    },
                },
            },
        },
        {
            "type": "custom",
            "data": {"event": "safety", "data": {"level": "supportive"}},
        },
        {
            "type": "messages",
            "data": (
                AIMessageChunk(content="分析内部文本"),
                {"langgraph_node": "analyze_emotion"},
            ),
        },
        {
            "type": "messages",
            "data": (
                AIMessageChunk(content="你好"),
                {"langgraph_node": "generate_reply"},
            ),
        },
    )

    events = [event async for event in adapt_graph_stream(chunks)]

    assert [event["event"] for event in events] == [
        "emotion_start",
        "emotion_done",
        "safety",
        "token",
    ]
    assert events[-1]["data"]["content"] == "你好"
    assert "safety_level" not in events[1]["data"]["state"]


@pytest.mark.asyncio
async def test_event_adapter_passes_only_the_stable_public_event_contract():
    public_names = [
        "run_started",
        "user_message",
        "emotion_start",
        "emotion_done",
        "emotion_error",
        "safety",
        "token",
        "done",
        "error",
    ]
    chunks = _chunks(
        *(
            {"type": "custom", "data": {"event": name, "data": {"name": name}}}
            for name in public_names
        ),
        {"type": "custom", "data": {"event": "internal_trace", "data": {}}},
        {"type": "custom", "data": {"event": "done", "data": "invalid"}},
    )

    events = [event async for event in adapt_graph_stream(chunks)]

    assert [event["event"] for event in events] == public_names


@pytest.mark.asyncio
async def test_event_adapter_accepts_visible_regeneration_chunks_only():
    chunks = _chunks(
        {
            "type": "messages",
            "data": (
                AIMessageChunk(content="新回复"),
                {"langgraph_node": "generate_variant"},
            ),
        },
        {
            "type": "messages",
            "data": (
                AIMessageChunk(content="危机模型内部流"),
                {"langgraph_node": "generate_crisis_reply"},
            ),
        },
    )

    events = [event async for event in adapt_graph_stream(chunks)]

    assert events == [{"event": "token", "data": {"content": "新回复"}}]


@pytest.mark.asyncio
async def test_sse_cancellation_closes_the_graph_iterator():
    closed = asyncio.Event()

    async def source():
        try:
            yield {
                "type": "custom",
                "data": {"event": "emotion_start", "data": {}},
            }
            await asyncio.Future()
        finally:
            closed.set()

    stream = web._sse_graph_events(
        source(),
        request_id=UUID(TURN_REQUEST_ID),
        operation="turn",
    )
    assert "event: run_started" in await anext(stream)
    assert "event: emotion_start" in await anext(stream)

    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert closed.is_set()


@pytest.mark.asyncio
async def test_sse_heartbeat_is_a_comment_not_a_public_event():
    async def source():
        await asyncio.sleep(0.02)
        yield {
            "type": "custom",
            "data": {"event": "done", "data": {"message_id": "ai-1", "content": "好"}},
        }

    stream = web._sse_graph_events(
        source(),
        request_id=UUID(TURN_REQUEST_ID),
        operation="turn",
        heartbeat_seconds=0.001,
    )

    assert "event: run_started" in await anext(stream)
    assert await anext(stream) == ": heartbeat\n\n"
    item = await anext(stream)
    while item.startswith(":"):
        item = await anext(stream)
    assert "event: done" in item
    await stream.aclose()


def test_turn_stream_uses_post_body_and_has_no_stream_id_registry(tmp_path, monkeypatch):
    model = SequenceModel("你好")
    with _client(tmp_path, monkeypatch, model) as (client, _):
        client_id, thread_id = _bootstrap(client)
        response = client.post(
            f"/api/clients/{client_id}/threads/{thread_id}/messages:stream",
            json={"message": "你好", "request_id": TURN_REQUEST_ID},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        assert events[0]["event"] == "run_started"
        assert events[-1]["event"] == "done"
        assert client.post("/api/chat/streams", json={"message": "旧接口"}).status_code == 404
        assert client.get("/api/chat/streams/expired").status_code == 404


def test_completed_turn_request_replays_without_models_or_duplicate_human(
    tmp_path, monkeypatch
):
    model = SequenceModel("只生成一次")
    with _client(tmp_path, monkeypatch, model) as (client, emotion_model):
        client_id, thread_id = _bootstrap(client)
        url = f"/api/clients/{client_id}/threads/{thread_id}/messages:stream"
        body = {"message": "请回答", "request_id": TURN_REQUEST_ID}

        first = client.post(url, json=body)
        calls = (model.calls, emotion_model.calls)
        replay = client.post(url, json=body)
        snapshot = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        ).json()

        assert first.status_code == replay.status_code == 200
        assert _events(replay.text)[-1]["data"]["replayed"] is True
        assert (model.calls, emotion_model.calls) == calls
        assert [message["id"] for message in snapshot["messages"]].count(
            f"human_{TURN_REQUEST_ID}"
        ) == 1


def test_failed_turn_emits_error_and_same_request_retries_without_partial_ai(
    tmp_path, monkeypatch
):
    model = SequenceModel(RuntimeError("provider unavailable"), "重试成功")
    with _client(tmp_path, monkeypatch, model) as (client, _):
        client_id, thread_id = _bootstrap(client)
        url = f"/api/clients/{client_id}/threads/{thread_id}/messages:stream"
        body = {"message": "请重试", "request_id": TURN_REQUEST_ID}

        failed = client.post(url, json=body)
        after_failure = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        ).json()
        retried = client.post(url, json=body)
        after_retry = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        ).json()

        assert _events(failed.text)[-1] == {
            "event": "error",
            "data": {"error_code": "stream_failed"},
        }
        failed_human_id = next(
            event["data"]["message_id"]
            for event in _events(failed.text)
            if event["event"] == "user_message"
        )
        assert all(item["role"] != "ai" for item in after_failure["messages"])
        assert _events(retried.text)[-1]["event"] == "done"
        assert [item["role"] for item in after_retry["messages"]] == ["human", "ai"]
        assert after_retry["messages"][0]["id"] == failed_human_id


def test_regeneration_stream_replaces_the_target_under_the_same_id(tmp_path, monkeypatch):
    model = SequenceModel("原回复", "新回复")
    with _client(tmp_path, monkeypatch, model) as (client, _):
        client_id, thread_id = _bootstrap(client)
        turn = client.post(
            f"/api/clients/{client_id}/threads/{thread_id}/messages:stream",
            json={"message": "你好", "request_id": TURN_REQUEST_ID},
        )
        target_id = _events(turn.text)[-1]["data"]["message_id"]

        regenerated = client.post(
            f"/api/clients/{client_id}/threads/{thread_id}/messages/{target_id}/regenerate:stream",
            json={"request_id": REGEN_REQUEST_ID, "reason": "其他"},
        )
        snapshot = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        ).json()
        calls_after_regeneration = model.calls
        replay = client.post(
            f"/api/clients/{client_id}/threads/{thread_id}/messages/{target_id}/regenerate:stream",
            json={"request_id": REGEN_REQUEST_ID, "reason": "其他"},
        )

        async def request_result():
            state = await client.app.state.graph_runtime.aget_state(client_id, thread_id)
            return state.values["processed_requests"][REGEN_REQUEST_ID]

        assert regenerated.status_code == 200
        assert _events(regenerated.text)[-1]["data"]["message_id"] == target_id
        assert _events(replay.text)[-1] == _events(regenerated.text)[-1]
        assert model.calls == calls_after_regeneration
        assert client.portal.call(request_result)["event_data"] == _events(
            regenerated.text
        )[-1]["data"]
        assert [item["id"] for item in snapshot["messages"]].count(target_id) == 1
        assert next(item for item in snapshot["messages"] if item["id"] == target_id)[
            "content"
        ] == "新回复"


def test_stream_preflight_errors_are_json_and_request_id_is_a_uuid(tmp_path, monkeypatch):
    model = SequenceModel("回复")
    with _client(tmp_path, monkeypatch, model) as (client, _):
        first_client, first_thread = _bootstrap(client)
        second_client, _ = _bootstrap(client)

        invalid_id = client.post(
            f"/api/clients/{first_client}/threads/{first_thread}/messages:stream",
            json={"message": "你好", "request_id": "req-1"},
        )
        unowned = client.post(
            f"/api/clients/{second_client}/threads/{first_thread}/messages:stream",
            json={"message": "你好", "request_id": TURN_REQUEST_ID},
        )
        invalid_client = client.post(
            f"/api/clients/tampered/threads/{first_thread}/messages:stream",
            json={"message": "你好", "request_id": TURN_REQUEST_ID},
        )

        assert invalid_id.status_code == 422
        assert unowned.status_code == 404
        assert unowned.headers["content-type"].startswith("application/json")
        assert invalid_client.status_code == 401
        assert invalid_client.headers["content-type"].startswith("application/json")
