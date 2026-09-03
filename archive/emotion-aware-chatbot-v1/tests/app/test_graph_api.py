from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

import chatbot.web as web
from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.emotion.feedback import load_emotion_feedback
from chatbot.graphs.runtime import build_graph_runtime


class StaticModel:
    async def ainvoke(self, value, config=None, **kwargs):
        return AIMessage(content="测试回复")

    def with_structured_output(self, schema):
        return ProfileModel()


class ProfileModel:
    async def ainvoke(self, value, config=None, **kwargs):
        return {"preferred_name": "小明", "response_style": "简短"}


@pytest.fixture
def graph_setup(tmp_path, monkeypatch):
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
    monkeypatch.setattr(web, "load_config", lambda argv: chat_config)
    monkeypatch.setattr(web, "load_graph_config", lambda: graph_config, raising=False)
    monkeypatch.setattr(
        web,
        "build_graph_runtime",
        lambda handles, chat, graph: build_graph_runtime(
            handles,
            chat,
            graph,
            model_factory=lambda config: (StaticModel(), StaticModel()),
            now=lambda: datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
        ),
        raising=False,
    )
    return graph_config


@pytest.fixture
def app_client(graph_setup):
    with TestClient(web.create_app()) as client:
        yield client


def bootstrap(client: TestClient) -> dict:
    response = client.post("/api/clients/bootstrap")
    assert response.status_code == 201
    return response.json()


def test_bootstrap_returns_signed_client_and_first_thread(app_client):
    payload = bootstrap(app_client)

    assert payload["client_id"].startswith("c_")
    raw = base64.urlsafe_b64decode(payload["client_id"][2:] + "==")
    assert len(raw) == 64
    assert payload["thread"]["thread_id"]


def test_bootstrap_client_token_rejects_tampering(app_client):
    payload = bootstrap(app_client)
    token = payload["client_id"]
    bad = token[:-1] + ("a" if token[-1] != "a" else "b")

    response = app_client.get(f"/api/clients/{bad}/threads")

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_client_id"


def test_thread_crud_is_client_scoped_and_newest_first(app_client):
    first = bootstrap(app_client)
    second_client = bootstrap(app_client)
    client_id = first["client_id"]
    first_thread = first["thread"]

    created = app_client.post(
        f"/api/clients/{client_id}/threads", json={"title": "第二个对话"}
    )
    assert created.status_code == 201
    second_thread = created.json()["thread"]

    listed = app_client.get(f"/api/clients/{client_id}/threads")
    assert listed.status_code == 200
    assert [item["thread_id"] for item in listed.json()["threads"]] == [
        second_thread["thread_id"],
        first_thread["thread_id"],
    ]
    for method in (app_client.get, app_client.delete):
        denied = method(
            f"/api/clients/{second_client['client_id']}/threads/{first_thread['thread_id']}"
        )
        assert denied.status_code == 404
        assert denied.json()["detail"] == "thread_not_found"

    deleted = app_client.delete(
        f"/api/clients/{client_id}/threads/{first_thread['thread_id']}"
    )
    assert deleted.status_code == 204
    assert app_client.get(
        f"/api/clients/{client_id}/threads/{first_thread['thread_id']}"
    ).status_code == 404


def test_delete_failure_keeps_store_record_until_saver_delete_succeeds(
    app_client, monkeypatch
):
    payload = bootstrap(app_client)
    client_id = payload["client_id"]
    thread_id = payload["thread"]["thread_id"]
    runtime = app_client.app.state.graph_runtime
    original_delete = runtime.checkpointer.adelete_thread

    async def fail_delete(value):
        raise OSError("checkpoint database unavailable")

    monkeypatch.setattr(runtime.checkpointer, "adelete_thread", fail_delete)
    failed = app_client.delete(f"/api/clients/{client_id}/threads/{thread_id}")
    assert failed.status_code == 500
    assert failed.json() == {"detail": "thread_delete_failed"}
    assert app_client.get(
        f"/api/clients/{client_id}/threads/{thread_id}"
    ).status_code == 200

    monkeypatch.setattr(runtime.checkpointer, "adelete_thread", original_delete)
    assert app_client.delete(
        f"/api/clients/{client_id}/threads/{thread_id}"
    ).status_code == 204


def test_thread_listing_reconciles_store_record_without_checkpoint(app_client):
    payload = bootstrap(app_client)
    client_id = payload["client_id"]
    runtime = app_client.app.state.graph_runtime

    async def create_stale_record():
        return await runtime.thread_repository.create(client_id, title="stale")

    stale = app_client.portal.call(create_stale_record)
    response = app_client.get(f"/api/clients/{client_id}/threads")

    assert response.status_code == 200
    assert stale.thread_id not in {
        record["thread_id"] for record in response.json()["threads"]
    }

    async def stale_removed():
        return await runtime.thread_repository.owns(client_id, stale.thread_id)

    assert app_client.portal.call(stale_removed) is False


def test_stale_store_record_stream_returns_json_404_before_sse_headers(app_client):
    payload = bootstrap(app_client)
    client_id = payload["client_id"]
    runtime = app_client.app.state.graph_runtime

    async def create_stale_record():
        return await runtime.thread_repository.create(client_id, title="stale-stream")

    stale = app_client.portal.call(create_stale_record)
    response = app_client.post(
        f"/api/clients/{client_id}/threads/{stale.thread_id}/messages:stream",
        json={"request_id": "00000000-0000-4000-8000-000000000099", "message": "你好"},
    )
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "thread_not_found"}


def test_thread_read_serializes_messages_and_emotion_metadata(app_client):
    payload = bootstrap(app_client)
    client_id = payload["client_id"]
    thread_id = payload["thread"]["thread_id"]
    runtime = app_client.app.state.graph_runtime

    async def seed_state():
        await runtime.graph.aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            {
                "messages": [
                    HumanMessage(id="human_1", content="你好", additional_kwargs={"turn_count": 1}),
                    AIMessage(
                        id="ai_1",
                        content="你好呀",
                        additional_kwargs={
                            "feedback": None,
                            "turn_count": 1,
                            "predicted_emotion": "content",
                            "safety_level": "normal",
                            "regeneration": {"reason": "其他"},
                            "original_content": "旧回复",
                            "regeneration_reason": "不准确",
                            "regenerated_at": "2026-09-01T10:00:00+00:00",
                            "regenerated": True,
                        },
                    ),
                ],
                "emotion_state": {"primary_emotion": "content", "safety_level": "normal"},
                "emotion_timeline": [
                    {"timestamp": "2026-09-01T10:00:00+00:00", "turn_count": 1,
                     "primary_emotion": "content", "safety_level": "normal"}
                ],
            },
            as_node="turn",
        )

    app_client.portal.call(seed_state)
    response = app_client.get(f"/api/clients/{client_id}/threads/{thread_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["messages"] == [
        {"role": "human", "content": "你好", "id": "human_1", "turn_count": 1},
        {
            "role": "ai", "content": "你好呀", "id": "ai_1", "feedback": None,
            "regeneration": {"reason": "其他"}, "turn_count": 1,
            "predicted_emotion": "content", "safety_level": "normal",
            "original_content": "旧回复", "regeneration_reason": "不准确",
            "regenerated_at": "2026-09-01T10:00:00+00:00", "regenerated": True,
        },
    ]
    assert body["emotion"]["primary_emotion"] == "content"
    assert body["emotion_timeline"][-1]["turn_count"] == 1
    assert body["metadata"]["thread_id"] == thread_id


def test_threads_and_messages_survive_lifespan_restart(graph_setup):
    app = web.create_app()
    with TestClient(app) as first_client:
        payload = bootstrap(first_client)
        client_id = payload["client_id"]
        thread_id = payload["thread"]["thread_id"]
        runtime = first_client.app.state.graph_runtime

        async def seed_message():
            await runtime.graph.aupdate_state(
                {"configurable": {"thread_id": thread_id}},
                {"messages": [HumanMessage(id="human_restart", content="保留我")]},
                as_node="turn",
            )

        first_client.portal.call(seed_message)

    with TestClient(web.create_app()) as second_client:
        response = second_client.get(f"/api/clients/{client_id}/threads/{thread_id}")
        assert response.status_code == 200
        assert response.json()["messages"][0]["content"] == "保留我"


def test_profile_is_client_scoped_and_draft_requires_owned_thread(app_client):
    first = bootstrap(app_client)
    second = bootstrap(app_client)
    client_id = first["client_id"]
    first_thread = first["thread"]["thread_id"]
    second_thread = app_client.post(f"/api/clients/{client_id}/threads", json={}).json()["thread"]["thread_id"]

    saved = app_client.put(
        f"/api/clients/{client_id}/profile",
        json={"profile": {"preferred_name": "小明", "unknown": "drop"}},
    )
    assert saved.status_code == 200
    assert app_client.get(f"/api/clients/{client_id}/profile").json()["profile"] == {
        "preferred_name": "小明"
    }
    assert app_client.get(f"/api/clients/{second['client_id']}/profile").json()["profile"] == {}
    assert app_client.get(f"/api/clients/{client_id}/threads/{first_thread}").status_code == 200
    assert app_client.get(f"/api/clients/{client_id}/threads/{second_thread}").status_code == 200

    assert app_client.post(
        f"/api/clients/{client_id}/profile/draft", json={"answers": []}
    ).status_code == 422
    denied = app_client.post(
        f"/api/clients/{client_id}/profile/draft",
        json={"thread_id": second["thread"]["thread_id"], "answers": []},
    )
    assert denied.status_code == 404
    draft = app_client.post(
        f"/api/clients/{client_id}/profile/draft",
        json={
            "thread_id": first_thread,
            "answers": [{"key": "preferred_name", "answer": "小明"}],
        },
    )
    assert draft.status_code == 200
    assert draft.json()["draft"] == {"preferred_name": "小明", "response_style": "简短"}
    assert app_client.get(f"/api/clients/{client_id}/profile").json()["profile"] == {
        "preferred_name": "小明"
    }


def test_message_and_emotion_feedback_use_owned_graph_state_and_store(app_client):
    first = bootstrap(app_client)
    second = bootstrap(app_client)
    client_id = first["client_id"]
    thread_id = first["thread"]["thread_id"]
    runtime = app_client.app.state.graph_runtime

    async def seed_ai():
        await runtime.graph.aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            {
                "messages": [AIMessage(id="ai_feedback", content="回复", additional_kwargs={"feedback": None})],
                "emotion_timeline": [{"timestamp": "2026-09-01T10:00:00+00:00", "turn_count": 1,
                                      "primary_emotion": "content", "safety_level": "normal"}],
            },
            as_node="turn",
        )

    app_client.portal.call(seed_ai)
    rated = app_client.patch(
        f"/api/clients/{client_id}/threads/{thread_id}/messages/ai_feedback/feedback",
        json={"feedback": "like"},
    )
    assert rated.status_code == 200
    assert rated.json() == {"message_id": "ai_feedback", "feedback": "like"}
    snapshot = app_client.get(
        f"/api/clients/{client_id}/threads/{thread_id}"
    ).json()
    assert next(
        message for message in snapshot["messages"] if message["id"] == "ai_feedback"
    )["feedback"] == "like"
    assert app_client.patch(
        f"/api/clients/{client_id}/threads/{thread_id}/messages/ai_feedback/feedback",
        json={"feedback": "dislike"},
    ).status_code == 409

    feedback_payload = {
        "feedback": "accurate", "message_id": "ai_feedback", "turn_count": 1,
        "predicted_emotion": "content", "corrected_emotion": "",
    }
    denied = app_client.post(
        f"/api/clients/{second['client_id']}/threads/{thread_id}/emotion-feedback",
        json=feedback_payload,
    )
    assert denied.status_code == 404
    saved = app_client.post(
        f"/api/clients/{client_id}/threads/{thread_id}/emotion-feedback",
        json=feedback_payload,
    )
    assert saved.status_code == 201
    timeline = app_client.get(
        f"/api/clients/{client_id}/threads/{thread_id}/emotion-timeline"
    )
    assert timeline.status_code == 200
    assert timeline.json()["timeline"][0]["primary_emotion"] == "content"

    async def stored_feedback():
        return await load_emotion_feedback(
            app_client.app.state.graph_store, client_id, thread_id
        )

    assert app_client.portal.call(stored_feedback)[0]["feedback"] == "accurate"


def test_message_feedback_rejects_invalid_value(app_client):
    payload = bootstrap(app_client)
    response = app_client.patch(
        "/api/clients/{}/threads/{}/messages/ai_missing/feedback".format(
            payload["client_id"], payload["thread"]["thread_id"]
        ),
        json={"feedback": "neutral"},
    )

    assert response.status_code == 422


def test_message_feedback_returns_stable_not_found(app_client):
    payload = bootstrap(app_client)
    response = app_client.patch(
        "/api/clients/{}/threads/{}/messages/ai_missing/feedback".format(
            payload["client_id"], payload["thread"]["thread_id"]
        ),
        json={"feedback": "like"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "message_not_found"}


def test_message_feedback_checkpoint_failure_has_stable_500(
    graph_setup, monkeypatch
):
    with TestClient(
        web.create_app(),
        raise_server_exceptions=False,
    ) as client:
        payload = bootstrap(client)
        client_id = payload["client_id"]
        thread_id = payload["thread"]["thread_id"]
        runtime = client.app.state.graph_runtime

        async def seed_ai():
            await runtime.graph.aupdate_state(
                {"configurable": {"thread_id": thread_id}},
                {"messages": [AIMessage(id="ai_write", content="回复")]},
                as_node="turn",
            )

        client.portal.call(seed_ai)

        async def fail_update(*args, **kwargs):
            raise OSError("checkpoint write failed")

        monkeypatch.setattr(runtime.graph, "aupdate_state", fail_update)
        response = client.patch(
            f"/api/clients/{client_id}/threads/{thread_id}/messages/ai_write/feedback",
            json={"feedback": "like"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal_error"}


def test_emotion_feedback_store_failure_has_stable_500(graph_setup, monkeypatch):
    async def fail_append(*args, **kwargs):
        raise OSError("store write failed")

    monkeypatch.setattr(web, "append_emotion_feedback", fail_append)
    with TestClient(
        web.create_app(),
        raise_server_exceptions=False,
    ) as client:
        payload = bootstrap(client)
        response = client.post(
            "/api/clients/{}/threads/{}/emotion-feedback".format(
                payload["client_id"], payload["thread"]["thread_id"]
            ),
            json={"feedback": "accurate"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal_error"}


def test_persistence_failure_has_stable_500_response(graph_setup, monkeypatch):
    async def fail_save(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(web, "save_profile", fail_save)
    with TestClient(
        web.create_app(),
        raise_server_exceptions=False,
    ) as client:
        payload = bootstrap(client)
        response = client.put(
            f"/api/clients/{payload['client_id']}/profile",
            json={"profile": {"preferred_name": "小明"}},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal_error"}
