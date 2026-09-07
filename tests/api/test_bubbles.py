from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from chatbot.llm.types import ModelDelta
from tests.api.helpers import parse_sse
from tests.emotion.helpers import create_app


class BubbleModel:
    parameters = {"model": "offline-bubbles"}

    def __init__(self, output, *, fail=False, pause=False):
        self.output = output
        self.fail = fail
        self.pause = pause
        self.release = asyncio.Event()
        self.calls = 0
        self.prompts = []

    async def stream(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        for index, part in enumerate(self.output):
            yield ModelDelta(content=part)
            if index == 0 and self.pause:
                await self.release.wait()
        if self.fail:
            raise RuntimeError("model unavailable")
        yield ModelDelta(finish_reason="stop")


def send(client, user_id, request_id):
    return client.post(f"/api/users/{user_id}/messages:stream", json={
        "request_id": request_id, "content": "今天有点累，想聊一会儿。",
    })


def test_bubbles_survive_history_and_idempotent_replay_without_protocol_in_context(app_config):
    """Catches merged bubbles on reload, JSON leakage, or a replay calling the model again."""
    model = BubbleModel(['{"messages":["今天挺熬人的啊。",', '"想聊什么都行。\\n我在听。"]}'])
    app = create_app(config=app_config, model=model)
    with TestClient(app) as client:
        user = client.post("/api/users/resolve", json={"identifier": "bubbles"}).json()["user"]["id"]
        request_id = str(uuid4())
        events = parse_sse(send(client, user, request_id).text)
        parts = [data["content"] for name, data in events if name == "bubble"]
        assert parts == ["今天挺熬人的啊。", "想聊什么都行。\n我在听。"]
        done = events[-1][1]["message"]
        assert done["bubbles"] == parts
        assert done["content"] == "今天挺熬人的啊。\n\n想聊什么都行。\n我在听。"
        history = client.get(f"/api/users/{user}/messages").json()["messages"]
        assert len(history) == 2
        assert history[-1]["bubbles"] == parts
        assert history[-1]["content"] == done["content"]
        replay = parse_sse(send(client, user, request_id).text)
        assert replay[-1][1]["message"]["bubbles"] == parts
        assert replay[-1][1]["replayed"] is True
        assert model.calls == 1
        send(client, user, str(uuid4()))
        assert model.prompts[-1][2].content == done["content"]
        assert all("bubbles_json" not in str(data) for _, data in events)


def test_first_bubble_is_persisted_and_delivered_before_model_finishes(app_config):
    """Catches buffering the whole response or publishing an uncommitted bubble."""
    model = BubbleModel(['{"messages":["先说一句。",', '"再接一句。"]}'], pause=True)
    app = create_app(config=app_config, model=model)
    with TestClient(app) as client:
        identity = client.post("/api/users/resolve", json={"identifier": "early"}).json()

        async def inspect_stream():
            stream = await app.state.coordinator.open_stream(identity["user"]["id"], str(uuid4()), "聊聊")
            async def first_bubble():
                async for event in stream.events():
                    if event.name == "bubble":
                        conversation = await app.state.conversations.get_default_by_user(identity["user"]["id"])
                        history = await app.state.messages.list_visible(conversation.id, limit=10)
                        assert history[-1]["status"] == "streaming"
                        assert history[-1]["bubbles"] == ["先说一句。"]
                        return event.data["content"]
            try:
                try:
                    first = await asyncio.wait_for(first_bubble(), timeout=1)
                except TimeoutError:
                    first = None
            finally:
                model.release.set()
                await app.state.coordinator.shutdown(2)
            return first
        assert client.portal.call(inspect_stream) == "先说一句。", "no bubble delivered before generation finished"


@pytest.mark.parametrize("tail,fail", [('"未完成', False), ('', True)])
def test_failed_generation_keeps_completed_bubbles_and_hides_partial_protocol(app_config, tail, fail):
    """Catches removing a sent bubble or showing half a JSON object after an error."""
    model = BubbleModel(['{"messages":["这条已经说完。",', tail], fail=fail)
    with TestClient(create_app(config=app_config, model=model)) as client:
        user = client.post("/api/users/resolve", json={"identifier": "partial"}).json()["user"]["id"]
        events = parse_sse(send(client, user, str(uuid4())).text)
        assert events[-1][0] == "error"
        assert [data["content"] for name, data in events if name == "bubble"] == ["这条已经说完。"]
        saved = client.get(f"/api/users/{user}/messages").json()["messages"][-1]
        assert saved["status"] == "failed"
        assert saved["bubbles"] == ["这条已经说完。"]
        assert saved["content"] == "这条已经说完。"


def test_configured_bubble_gap_reaches_stream_without_delaying_generation(app_config):
    """Catches frontend using its own gap instead of the configured value."""
    config = replace(app_config, chat_bubble_gap_ms=750)
    with TestClient(create_app(config=config, model=BubbleModel(['{"messages":["好呀。"]}']))) as client:
        user = client.post("/api/users/resolve", json={"identifier": "gap"}).json()["user"]["id"]
        events = parse_sse(send(client, user, str(uuid4())).text)
        assert events[0][1]["bubble_gap_ms"] == 750
