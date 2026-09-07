from __future__ import annotations

import asyncio
import json
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


def test_bubbles_preserve_display_text_and_use_protocol_in_model_context(app_config):
    """Keep display history intact while giving the next model a valid JSON example."""
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
        assert json.loads(model.prompts[-1][2].content) == {"messages": [done["content"]]}
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


class CorrectingBubbleModel(BubbleModel):
    parameters = {"model": "offline-bubbles", "response_format": {"type": "json_object"}}

    def __init__(self, replies):
        super().__init__([])
        self.replies = replies

    async def stream(self, prompt):
        self.prompts.append(list(prompt))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        yield ModelDelta(content=reply)
        yield ModelDelta(finish_reason="stop")


@pytest.mark.parametrize('invalid', ['["不能直接接受的数组"]', '不符合协议的纯文本'])
def test_invalid_reply_is_corrected_once_without_publishing_invalid_output(app_config, invalid):
    model = CorrectingBubbleModel([invalid, '{"messages":["重新生成的合规回复"]}'])
    with TestClient(create_app(config=app_config, model=model)) as client:
        user = client.post('/api/users/resolve', json={'identifier': 'correction'}).json()['user']['id']
        request = str(uuid4())
        events = parse_sse(send(client, user, request).text)
        assert events[-1][0] == 'done'
        assert [d['content'] for n, d in events if n == 'bubble'] == ['重新生成的合规回复']
        assert not any(n == 'token' and invalid in d.get('content', '') for n, d in events)
        assert model.calls == 2
        assert 'messages' in model.prompts[1][-1].content
        replay = parse_sse(send(client, user, request).text)
        assert replay[-1][1]['replayed'] is True
        assert model.calls == 2
        import sqlite3, json
        with sqlite3.connect(app_config.sqlite_db_path) as db:
            trace = json.loads(db.execute("select trace_json from messages where role='assistant'").fetchone()[0])
        assert trace['reply_attempts'][0]['raw_output'] == invalid
        assert trace['reply_attempts'][0]['error_code'] == 'reply_format_error'
        assert len(trace['model_calls']) == 2


def test_repeated_invalid_reply_stops_with_specific_error(app_config):
    model = CorrectingBubbleModel(['["仍然不合规"]'])
    with TestClient(create_app(config=app_config, model=model)) as client:
        user = client.post('/api/users/resolve', json={'identifier': 'bad-format'}).json()['user']['id']
        events = parse_sse(send(client, user, str(uuid4())).text)
        assert events[-1][0] == 'error'
        assert events[-1][1]['code'] == 'reply_format_error'
        assert model.calls == 2
        assert not any(n in {'token', 'bubble'} for n, _ in events)


def test_format_failure_after_published_bubble_never_retries(app_config):
    model = CorrectingBubbleModel(['{"messages":["已经显示",false]}'])
    with TestClient(create_app(config=app_config, model=model)) as client:
        user = client.post('/api/users/resolve', json={'identifier': 'no-duplicate'}).json()['user']['id']
        events = parse_sse(send(client, user, str(uuid4())).text)
        assert events[-1][0] == 'error'
        assert model.calls == 1
        assert [d['content'] for n, d in events if n == 'bubble'] == ['已经显示']


@pytest.mark.parametrize('truncated', [False, True])
def test_transport_failure_or_token_limit_does_not_trigger_format_retry(app_config, truncated):
    class FailedModel(CorrectingBubbleModel):
        async def stream(self, prompt):
            self.calls += 1
            if truncated:
                yield ModelDelta(content='["incomplete', finish_reason='length')
            else:
                raise RuntimeError('connection failed')
    model = FailedModel([])
    with TestClient(create_app(config=app_config, model=model)) as client:
        user = client.post('/api/users/resolve', json={'identifier': 'no-retry'}).json()['user']['id']
        events = parse_sse(send(client, user, str(uuid4())).text)
        assert events[-1][0] == 'error'
        assert events[-1][1]['code'] == 'model_error'
        assert model.calls == 1
