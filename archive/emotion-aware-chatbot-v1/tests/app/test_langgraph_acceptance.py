"""Deterministic end-to-end acceptance coverage for the LangGraph runtime."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import chatbot.web as web
from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.graphs.runtime import RuntimeOperationError, build_graph_runtime
from chatbot.memory import MemoryCandidate
from chatbot.persistence.runtime import open_persistence
from chatbot.persistence.threads import ThreadRepository
from chatbot.profile import load_profile, save_profile


NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
NORMAL_EMOTION = (
    '{"primary_emotion":"content","confidence":0.8,'
    '"secondary_emotions":[],"evidence":"steady",'
    '"reply_strategy":"respond naturally","trajectory_note":"stable",'
    '"safety_level":"normal"}'
)


class RecordingModel:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses) or ["测试回复"]
        self.calls = []

    async def ainvoke(self, value, config=None, **kwargs):
        self.calls.append(value)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)

    def with_structured_output(self, schema):
        return self


def configs(tmp_path, *, interval: int = 5) -> tuple[ChatConfig, GraphConfig]:
    llm = LlmConfig(provider="test", api_key="test", model="test", temperature=0)
    return (
        ChatConfig(chat_llm=llm, emotion_llm=llm, emotion_interval=interval),
        GraphConfig(
            checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
            store_db_path=str(tmp_path / "store.sqlite3"),
            timeline_limit=50,
            request_history_limit=64,
            strict_msgpack=True,
            client_id_signing_secret="acceptance-signing-secret-32-bytes-minimum",
        ),
    )


async def consume(stream) -> list[dict]:
    return [part async for part in stream]


async def consume_until_error(stream) -> tuple[list[dict], Exception | None]:
    parts = []
    try:
        async for part in stream:
            parts.append(part)
    except Exception as exc:
        return parts, exc
    return parts, None


def user_message_events(parts: list[dict]) -> list[dict]:
    return [
        part["data"]
        for part in parts
        if part.get("type") == "custom"
        and isinstance(part.get("data"), dict)
        and part["data"].get("event") == "user_message"
    ]


async def turn(runtime, client_id: str, thread_id: str, message: str, request_id=None):
    request_id = request_id or str(uuid4())
    return await consume(
        await runtime.astream_turn(client_id, thread_id, request_id, message)
    )


@pytest.mark.asyncio
async def test_first_turn_emotion_and_done_use_real_sqlite_graph(tmp_path):
    chat, graph = configs(tmp_path)
    chat_model = RecordingModel("我在听。")
    emotion_model = RecordingModel(NORMAL_EMOTION)
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles,
            chat,
            graph,
            model_factory=lambda _: (chat_model, emotion_model),
            now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")

        parts = await turn(runtime, "client-a", record.thread_id, "今天很平静", "req-1")
        state = await runtime.aget_state("client-a", record.thread_id)

        custom = [part["data"]["event"] for part in parts if part["type"] == "custom"]
        assert custom == ["user_message", "emotion_start", "emotion_done", "safety", "done"]
        assert [message.type for message in state.values["messages"]] == ["human", "ai"]
        assert state.values["emotion_state"]["primary_emotion"] == "content"


@pytest.mark.asyncio
async def test_client_context_shared_across_threads_but_thread_state_isolated(tmp_path):
    chat, graph = configs(tmp_path)
    chat_model = RecordingModel("第一条", "第二条")
    emotion_model = RecordingModel(NORMAL_EMOTION)
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles,
            chat,
            graph,
            model_factory=lambda _: (chat_model, emotion_model),
            now=lambda: NOW,
        )
        first = await runtime.acreate_thread("client-a")
        second = await runtime.acreate_thread("client-a")
        foreign = await runtime.acreate_thread("client-b")
        await save_profile(handles.store, "client-a", {"preferred_name": "小明"})
        await runtime.memory_repository.aremember(
            "client-a",
            [MemoryCandidate(content="用户喜欢爵士乐。", category="preference")],
        )

        await turn(runtime, "client-a", first.thread_id, "我喜欢爵士乐", "req-a")
        await turn(runtime, "client-a", second.thread_id, "推荐爵士乐", "req-b")
        first_state = await runtime.aget_state("client-a", first.thread_id)
        second_state = await runtime.aget_state("client-a", second.thread_id)

        second_prompt = str(chat_model.calls[-1])
        assert "小明" in second_prompt and "爵士乐" in second_prompt
        assert [message.id for message in first_state.values["messages"]] == [
            "human_req-a", "ai_req-a"
        ]
        assert [message.id for message in second_state.values["messages"]] == [
            "human_req-b", "ai_req-b"
        ]
        assert len(first_state.values["emotion_timeline"]) == 1
        assert len(second_state.values["emotion_timeline"]) == 1
        assert await load_profile(handles.store, "client-b") == {}
        assert await runtime.memory_repository.asearch("client-b", "爵士乐", limit=5) == []
        with pytest.raises(RuntimeOperationError, match="thread_not_found"):
            await runtime.aget_state("client-b", first.thread_id)
        assert foreign.thread_id != first.thread_id


@pytest.mark.asyncio
async def test_interval_reuse_risk_force_and_crisis_monotonicity(tmp_path):
    chat, graph = configs(tmp_path, interval=5)
    chat_model = RecordingModel("普通回复", "普通回复", "危机支持")
    emotion_model = RecordingModel(NORMAL_EMOTION)
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (chat_model, emotion_model), now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")
        await turn(runtime, "client-a", record.thread_id, "第一轮", "req-1")
        await turn(runtime, "client-a", record.thread_id, "普通第二轮", "req-2")
        assert len(emotion_model.calls) == 1
        await turn(runtime, "client-a", record.thread_id, "我现在打算自杀", "req-3")
        state = await runtime.aget_state("client-a", record.thread_id)
        assert len(emotion_model.calls) == 2
        assert state.values["messages"][-1].additional_kwargs["safety_level"] == "crisis"


@pytest.mark.asyncio
async def test_regeneration_replaces_the_same_ai_message(tmp_path):
    chat, graph = configs(tmp_path)
    chat_model = RecordingModel("原回复", "新回复")
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (chat_model, RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")
        await turn(runtime, "client-a", record.thread_id, "你好", "req-1")
        with pytest.raises(RuntimeOperationError) as conflict_info:
            await runtime.astream_turn(
                "client-a", record.thread_id, "req-1", "不同正文"
            )
        assert conflict_info.value.code == "request_id_conflict"
        with pytest.raises(RuntimeOperationError) as cross_operation_info:
            await runtime.astream_regeneration(
                "client-a", record.thread_id, "req-1", "ai_req-1", "不准确"
            )
        assert cross_operation_info.value.code == "request_id_conflict"
        await consume(await runtime.astream_regeneration(
            "client-a", record.thread_id, "regen-1", "ai_req-1", "不准确"
        ))
        replay = await consume(await runtime.astream_regeneration(
            "client-a", record.thread_id, "regen-1", "ai_req-1", "不准确"
        ))
        assert any(
            part.get("type") == "custom"
            and part.get("data", {}).get("data", {}).get("replayed") is True
            for part in replay
        )
        with pytest.raises(RuntimeOperationError) as regen_conflict_info:
            await runtime.astream_regeneration(
                "client-a", record.thread_id, "regen-1", "ai_req-1", "其他"
            )
        assert regen_conflict_info.value.code == "request_id_conflict"
        with pytest.raises(RuntimeOperationError) as target_conflict_info:
            await runtime.astream_regeneration(
                "client-a", record.thread_id, "regen-1", "ai-other", "不准确"
            )
        assert target_conflict_info.value.code == "request_id_conflict"
        state = await runtime.aget_state("client-a", record.thread_id)
        ai_messages = [m for m in state.values["messages"] if m.type == "ai"]
        assert [(m.id, m.content) for m in ai_messages] == [("ai_req-1", "新回复")]
        assert ai_messages[0].additional_kwargs["regenerated"] is True
        assert ai_messages[0].additional_kwargs["regeneration_reason"] == "不准确"


@pytest.mark.asyncio
async def test_restart_restores_state_and_thread_order(tmp_path):
    chat, graph = configs(tmp_path)
    client_id = "client-a"
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (RecordingModel("回复"), RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        )
        first = await runtime.acreate_thread(client_id)
        second = await runtime.acreate_thread(client_id)
        await turn(runtime, client_id, first.thread_id, "保留", "req-1")

    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (RecordingModel("unused"), RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        )
        records = await runtime.alist_threads(client_id)
        state = await runtime.aget_state(client_id, first.thread_id)
        assert [r.thread_id for r in records] == [first.thread_id, second.thread_id]
        assert state.values["messages"][-1].content == "回复"


@pytest.mark.asyncio
async def test_delete_is_saver_before_store_and_cross_client_timeline_is_404(
    tmp_path, monkeypatch
):
    chat, graph = configs(tmp_path)
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (RecordingModel(), RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")
        calls = []
        saver_delete = runtime.checkpointer.adelete_thread
        store_delete = runtime.thread_repository.delete_record

        async def tracked_saver(thread_id):
            calls.append("saver")
            await saver_delete(thread_id)

        async def tracked_store(client_id, thread_id):
            calls.append("store")
            await store_delete(client_id, thread_id)

        monkeypatch.setattr(runtime.checkpointer, "adelete_thread", tracked_saver)
        monkeypatch.setattr(runtime.thread_repository, "delete_record", tracked_store)
        await runtime.adelete_thread("client-a", record.thread_id)
        assert calls == ["saver", "store"]

    monkeypatch.setattr(web, "load_config", lambda argv: chat)
    monkeypatch.setattr(web, "load_graph_config", lambda: graph)
    monkeypatch.setattr(
        web, "build_graph_runtime",
        lambda handles, chat_config, graph_config: build_graph_runtime(
            handles, chat_config, graph_config,
            model_factory=lambda _: (RecordingModel(), RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        ),
    )
    with TestClient(web.create_app()) as client:
        owner = client.post("/api/clients/bootstrap").json()
        foreign = client.post("/api/clients/bootstrap").json()
        response = client.get(
            f"/api/clients/{foreign['client_id']}/threads/"
            f"{owner['thread']['thread_id']}/emotion-timeline"
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "thread_not_found"}


@pytest.mark.asyncio
async def test_store_write_failure_warns_without_losing_reply(
    tmp_path, monkeypatch, caplog
):
    chat, graph = configs(tmp_path)
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (RecordingModel("回复仍完成"), RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")

        async def fail_write(*args, **kwargs):
            raise OSError("store unavailable")

        monkeypatch.setattr(runtime.memory_repository, "aremember", fail_write)
        await turn(runtime, "client-a", record.thread_id, "我喜欢爵士乐", "req-1")
        state = await runtime.aget_state("client-a", record.thread_id)
        assert state.values["messages"][-1].content == "回复仍完成"
        assert state.values["processed_requests"]["req-1"]["status"] == "completed"
        assert "memory extraction write failed" in caplog.text


@pytest.mark.asyncio
async def test_generation_failure_has_no_partial_ai_and_same_request_retries(tmp_path):
    chat, graph = configs(tmp_path)
    chat_model = RecordingModel(RuntimeError("provider unavailable"), "重试成功")
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (chat_model, RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")
        failed_parts, error = await consume_until_error(
            await runtime.astream_turn(
                "client-a", record.thread_id, "same-request", "你好"
            )
        )
        assert isinstance(error, RuntimeError)
        assert str(error) == "provider unavailable"
        assert len(user_message_events(failed_parts)) == 1
        failed = await runtime.aget_state("client-a", record.thread_id)
        assert [(m.id, m.type, m.content) for m in failed.values["messages"]] == [
            ("human_same-request", "human", "你好")
        ]
        assert failed.values["turn_count"] == 1

    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (RecordingModel("重试成功"), RecordingModel(NORMAL_EMOTION)),
            now=lambda: NOW,
        )
        retry_parts = await turn(
            runtime, "client-a", record.thread_id, "你好", "same-request"
        )
        assert user_message_events(retry_parts) == []
        recovered = await runtime.aget_state("client-a", record.thread_id)
        assert [m.id for m in recovered.values["messages"]] == [
            "human_same-request", "ai_same-request"
        ]
        assert recovered.values["turn_count"] == 1
        assert recovered.values["messages"][0].content == "你好"
        assert recovered.values["messages"][-1].content == "重试成功"


@pytest.mark.asyncio
async def test_legacy_thread_sort_fallback_survives_same_second_sqlite_reopen(tmp_path):
    _, graph = configs(tmp_path)
    timestamp = NOW.isoformat(timespec="seconds")
    namespace = ("client-a", "threads")
    async with open_persistence(graph) as handles:
        # Create in reverse lexical order: Store timestamps, not UUID/key order,
        # must identify the later legacy record.
        for index, thread_id in enumerate(("legacy-z", "legacy-a")):
            record_timestamp = timestamp.replace("+00:00", "Z") if index == 0 else timestamp
            await handles.store.aput(namespace, thread_id, {
                "thread_id": thread_id,
                "title": thread_id,
                "created_at": record_timestamp,
                "updated_at": record_timestamp,
            })
            if index == 0:
                # Official SQLite Store timestamps are second-resolution. Both
                # legacy record timestamps remain the same second, while this
                # makes the Store write-time fallback observable after reopen.
                await asyncio.sleep(1.05)

    async with open_persistence(graph) as handles:
        records = await ThreadRepository(handles.store, now=lambda: NOW).list("client-a")
        assert [record.thread_id for record in records] == ["legacy-a", "legacy-z"]


@pytest.mark.asyncio
async def test_current_same_second_thread_order_survives_sqlite_reopen(tmp_path):
    _, graph = configs(tmp_path)
    async with open_persistence(graph) as handles:
        repository = ThreadRepository(handles.store, now=lambda: NOW)
        first = await repository.create("client-a", title="first")
        second = await repository.create("client-a", title="second")

    async with open_persistence(graph) as handles:
        records = await ThreadRepository(handles.store, now=lambda: NOW).list("client-a")
        assert [record.thread_id for record in records] == [
            second.thread_id,
            first.thread_id,
        ]


@pytest.mark.asyncio
async def test_record_timestamp_is_canonical_before_store_write_time(tmp_path):
    _, graph = configs(tmp_path)
    namespace = ("client-c", "threads")
    async with open_persistence(graph) as handles:
        # Newer business timestamp is written first; later Store write must not
        # overtake it when the record timestamps differ.
        await handles.store.aput(namespace, "business-new", {
            "thread_id": "business-new",
            "title": "new",
            "created_at": "2026-09-01T10:00:01Z",
            "updated_at": "2026-09-01T10:00:01Z",
        })
        await asyncio.sleep(1.05)
        await handles.store.aput(namespace, "business-old", {
            "thread_id": "business-old",
            "title": "old",
            "created_at": "2026-09-01T10:00:00+00:00",
            "updated_at": "2026-09-01T10:00:00+00:00",
        })

    async with open_persistence(graph) as handles:
        records = await ThreadRepository(handles.store, now=lambda: NOW).list("client-c")
        assert [record.thread_id for record in records] == [
            "business-new", "business-old"
        ]


@pytest.mark.asyncio
async def test_user_message_event_is_request_scoped_across_failure_interleaving_and_cancel(
    tmp_path,
):
    chat, graph = configs(tmp_path)
    chat_model = RecordingModel(
        RuntimeError("A failed"),
        "A recovered",
        "B completed",
        "cancel recovered",
    )
    emotion_model = RecordingModel(NORMAL_EMOTION)
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (chat_model, emotion_model),
            now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")

        a_first, a_error = await consume_until_error(
            await runtime.astream_turn(
                "client-a", record.thread_id, "request-a", "原始 A"
            )
        )
        assert isinstance(a_error, RuntimeError)
        assert len(user_message_events(a_first)) == 1
        failed_state = await runtime.aget_state("client-a", record.thread_id)
        assert failed_state.values["pending_turn"]["request_id"] == "request-a"

        with pytest.raises(RuntimeOperationError) as b_error:
            await runtime.astream_turn(
                "client-a", record.thread_id, "request-b", "B 输入"
            )
        assert b_error.value.code == "turn_in_progress"
        assert all("B 输入" not in str(prompt) for prompt in emotion_model.calls)

        a_retry = await turn(
            runtime, "client-a", record.thread_id, "原始 A", "request-a"
        )
        assert user_message_events(a_retry) == []

        a_replay = await turn(
            runtime, "client-a", record.thread_id, "原始 A", "request-a"
        )
        assert user_message_events(a_replay) == []

        b_parts = await turn(
            runtime, "client-a", record.thread_id, "B 输入", "request-b"
        )
        assert len(user_message_events(b_parts)) == 1

        cancelled = await runtime.astream_turn(
            "client-a", record.thread_id, "request-cancel", "取消后保留"
        )
        first_part = await anext(cancelled)
        assert len(user_message_events([first_part])) == 1
        await cancelled.aclose()

        cancel_retry = await turn(
            runtime,
            "client-a",
            record.thread_id,
            "取消后保留",
            "request-cancel",
        )
        assert user_message_events(cancel_retry) == []
        state = await runtime.aget_state("client-a", record.thread_id)
        human = [message for message in state.values["messages"] if message.type == "human"]
        assert [(message.id, message.content) for message in human] == [
            ("human_request-a", "原始 A"),
            ("human_request-b", "B 输入"),
            ("human_request-cancel", "取消后保留"),
        ]
        assert state.values["turn_count"] == 3


@pytest.mark.asyncio
async def test_turn_open_atomically_reserves_pending_without_holding_lock(tmp_path):
    chat, graph = configs(tmp_path)
    async with open_persistence(graph) as handles:
        runtime = build_graph_runtime(
            handles, chat, graph,
            model_factory=lambda _: (
                RecordingModel("A completed"), RecordingModel(NORMAL_EMOTION)
            ),
            now=lambda: NOW,
        )
        record = await runtime.acreate_thread("client-a")

        delayed_a = await runtime.astream_turn(
            "client-a", record.thread_id, "request-a", "A input"
        )
        state = await runtime.aget_state("client-a", record.thread_id)
        assert state.values["pending_turn"]["request_id"] == "request-a"
        assert runtime.lock_for(record.thread_id).locked() is False

        with pytest.raises(RuntimeOperationError) as blocked:
            await runtime.astream_turn(
                "client-a", record.thread_id, "request-b", "B must not enter"
            )
        assert blocked.value.code == "turn_in_progress"

        exact_retry = await runtime.astream_turn(
            "client-a", record.thread_id, "request-a", "A input"
        )
        retry_parts = await consume(exact_retry)
        assert user_message_events(retry_parts) == []
        assert runtime.lock_for(record.thread_id).locked() is False
        await delayed_a.aclose()
        final = await runtime.aget_state("client-a", record.thread_id)
        assert [message.id for message in final.values["messages"]] == [
            "human_request-a", "ai_request-a"
        ]


def _install_web_runtime(monkeypatch, chat, graph, chat_model, emotion_model):
    monkeypatch.setattr(web, "load_config", lambda argv: chat)
    monkeypatch.setattr(web, "load_graph_config", lambda: graph)
    monkeypatch.setattr(
        web,
        "build_graph_runtime",
        lambda handles, chat_config, graph_config: build_graph_runtime(
            handles,
            chat_config,
            graph_config,
            model_factory=lambda _: (chat_model, emotion_model),
            now=lambda: NOW,
        ),
    )


def test_real_lifespan_http_flow_and_restart_persistence(tmp_path, monkeypatch):
    chat, graph = configs(tmp_path)
    _install_web_runtime(
        monkeypatch,
        chat,
        graph,
        RecordingModel("HTTP 回复"),
        RecordingModel(NORMAL_EMOTION),
    )
    with TestClient(web.create_app()) as client:
        bootstrap = client.post("/api/clients/bootstrap").json()
        client_id = bootstrap["client_id"]
        thread_id = bootstrap["thread"]["thread_id"]
        assert client.put(
            f"/api/clients/{client_id}/profile",
            json={"profile": {"preferred_name": "小明"}},
        ).status_code == 200
        stream = client.post(
            f"/api/clients/{client_id}/threads/{thread_id}/messages:stream",
            json={"request_id": str(uuid4()), "message": "第一轮"},
        )
        assert stream.status_code == 200
        assert stream.text.count("event: user_message") == 1
        assert "event: emotion_done" in stream.text
        assert "event: safety" in stream.text
        assert "event: done" in stream.text
        snapshot = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        ).json()
        ai_id = snapshot["messages"][-1]["id"]
        assert client.get(
            f"/api/clients/{client_id}/threads/{thread_id}/emotion-timeline"
        ).json()["timeline"]

    _install_web_runtime(
        monkeypatch,
        chat,
        graph,
        RecordingModel("unused"),
        RecordingModel(NORMAL_EMOTION),
    )
    with TestClient(web.create_app()) as client:
        snapshot = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        )
        assert snapshot.status_code == 200
        assert snapshot.json()["messages"][-1]["id"] == ai_id
        assert client.get(f"/api/clients/{client_id}/profile").json()["profile"] == {
            "preferred_name": "小明"
        }


def test_api_failed_stream_keeps_human_and_retry_does_not_duplicate(
    tmp_path, monkeypatch
):
    chat, graph = configs(tmp_path)
    chat_model = RecordingModel(RuntimeError("provider unavailable"), "API 重试成功")
    _install_web_runtime(
        monkeypatch, chat, graph, chat_model, RecordingModel(NORMAL_EMOTION)
    )
    request_id = str(uuid4())
    with TestClient(web.create_app()) as client:
        bootstrap = client.post("/api/clients/bootstrap").json()
        client_id = bootstrap["client_id"]
        thread_id = bootstrap["thread"]["thread_id"]
        endpoint = f"/api/clients/{client_id}/threads/{thread_id}/messages:stream"

        failed = client.post(
            endpoint, json={"request_id": request_id, "message": "保存这条输入"}
        )
        assert "event: error" in failed.text
        assert failed.text.count("event: user_message") == 1
        assert '"error_code": "stream_failed"' in failed.text
        failed_snapshot = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        ).json()
        assert [(m["id"], m["role"]) for m in failed_snapshot["messages"]] == [
            (f"human_{request_id}", "human")
        ]
        assert failed_snapshot["pending_turn"] == {
            "request_id": request_id,
            "content": "保存这条输入",
        }
        blocked = client.post(
            endpoint,
            json={"request_id": str(uuid4()), "message": "B 不应进入"},
        )
        assert blocked.status_code == 409
        assert blocked.json() == {"detail": "turn_in_progress"}

        recovered = client.post(
            endpoint, json={"request_id": request_id, "message": "保存这条输入"}
        )
        assert "event: done" in recovered.text
        assert "event: user_message" not in recovered.text
        snapshot = client.get(
            f"/api/clients/{client_id}/threads/{thread_id}"
        ).json()
        assert [(m["id"], m["role"]) for m in snapshot["messages"]] == [
            (f"human_{request_id}", "human"),
            (f"ai_{request_id}", "ai"),
        ]
        assert snapshot["messages"][0]["content"] == "保存这条输入"
        assert snapshot["pending_turn"] is None
