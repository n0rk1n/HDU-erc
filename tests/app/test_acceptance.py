from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import empty_checkpoint
from pydantic import SecretStr

from chatbot.core.config import AppConfig
from chatbot.llm.types import ModelDelta, TokenUsage
from tests.emotion.helpers import create_app
from tests.api.helpers import parse_sse


class DeterministicAcceptanceModel:
    """Offline adapter that records complete prompts without network access."""

    provider = "acceptance-offline"
    parameters = {"model": "acceptance-offline", "temperature": 0}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.prompts: list[list[tuple[str, str]]] = []

    async def stream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[ModelDelta]:
        self.calls += 1
        self.prompts.append(
            [(message.type, str(message.content)) for message in prompt]
        )
        current_content = str(prompt[-1].content)
        if self.fail:
            yield ModelDelta(content="部分回复", reasoning="失败前依据")
            raise RuntimeError("deterministic offline failure")
        yield ModelDelta(content="答复：")
        yield ModelDelta(
            content=current_content,
            reasoning=f"依据：{current_content}",
            usage=TokenUsage(input_tokens=7, output_tokens=3, total_tokens=10),
            finish_reason="stop",
            response_metadata={"request_id": f"offline-{self.calls}"},
        )


def _config(database_path: Path) -> AppConfig:
    return AppConfig(
        llm_api_key=SecretStr("offline-placeholder"),
        llm_model="acceptance-offline",
        llm_base_url=None,
        llm_temperature=0,
        llm_timeout_seconds=1,
        context_message_limit=40,
        sqlite_db_path=database_path,
    )


def _resolve(client: TestClient, identifier: str) -> dict[str, object]:
    response = client.post("/api/users/resolve", json={"identifier": identifier})
    assert response.status_code == 200
    return response.json()


def _stream(
    client: TestClient,
    user_id: int,
    content: str,
    *,
    request_id: str | None = None,
) -> list[tuple[str, dict[str, object]]]:
    response = client.post(
        f"/api/users/{user_id}/messages:stream",
        json={"request_id": request_id or str(uuid4()), "content": content},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    return parse_sse(response.text)


def _prompt_text(prompt: list[tuple[str, str]]) -> str:
    return "\n".join(content for _, content in prompt)


def test_web_exports_lazy_fastapi_app_without_requiring_credentials(tmp_path) -> None:
    """Catches the documented ASGI import constructing config/model before lifespan."""
    environment = os.environ.copy()
    environment.pop("LLM_API_KEY", None)
    repository_root = Path(__file__).resolve().parents[2]
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(repository_root), inherited_pythonpath)
        if item is not None
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fastapi import FastAPI; "
                "from chatbot.web import app as web_app; "
                "from chatbot.main import app as main_app; "
                "assert isinstance(web_app, FastAPI); "
                "assert isinstance(main_app, FastAPI)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_two_users_keep_business_history_and_model_context_isolated(tmp_path) -> None:
    """Catches conversation lookup mixing one user's persisted or model context."""
    model = DeterministicAcceptanceModel()
    app = create_app(config=_config(tmp_path / "acceptance.sqlite3"), model=model)

    with TestClient(app) as client:
        alice = _resolve(client, "acceptance-alice")
        bob = _resolve(client, "acceptance-bob")
        alice_id = int(alice["user"]["id"])
        bob_id = int(bob["user"]["id"])

        assert _stream(client, alice_id, "Alice secret")[-1][0] == "done"
        assert _stream(client, bob_id, "Bob secret")[-1][0] == "done"
        assert _stream(client, alice_id, "Alice follow-up")[-1][0] == "done"
        assert _stream(client, bob_id, "Bob follow-up")[-1][0] == "done"

        alice_history = client.get(f"/api/users/{alice_id}/messages").json()[
            "messages"
        ]
        bob_history = client.get(f"/api/users/{bob_id}/messages").json()["messages"]

    assert [item["content"] for item in alice_history] == [
        "Alice secret",
        "答复：Alice secret",
        "Alice follow-up",
        "答复：Alice follow-up",
    ]
    assert [item["content"] for item in bob_history] == [
        "Bob secret",
        "答复：Bob secret",
        "Bob follow-up",
        "答复：Bob follow-up",
    ]
    assert model.calls == 4
    alice_prompts = [_prompt_text(model.prompts[index]) for index in (0, 2)]
    bob_prompts = [_prompt_text(model.prompts[index]) for index in (1, 3)]
    assert all("Bob secret" not in prompt for prompt in alice_prompts)
    assert all("Alice secret" not in prompt for prompt in bob_prompts)
    assert "Alice secret" in alice_prompts[1]
    assert "答复：Alice secret" in alice_prompts[1]
    assert "Bob secret" in bob_prompts[1]
    assert "答复：Bob secret" in bob_prompts[1]


def test_completed_and_failed_request_replay_never_calls_model_again(tmp_path) -> None:
    """Catches terminal request replay accidentally entering model generation again."""
    database_path = tmp_path / "replay.sqlite3"
    completed_model = DeterministicAcceptanceModel()
    completed_app = create_app(
        config=_config(database_path), model=completed_model
    )
    completed_request = str(uuid4())

    with TestClient(completed_app) as client:
        user_id = int(_resolve(client, "completed-replay")["user"]["id"])
        first = _stream(
            client, user_id, "same completed request", request_id=completed_request
        )
        replay = _stream(
            client, user_id, "same completed request", request_id=completed_request
        )

    assert first[-1] == (
        "done",
        {"message": first[-1][1]["message"], "replayed": False},
    )
    assert [event for event, _ in replay] == ["run_started", "done"]
    assert replay[-1][1]["replayed"] is True
    assert completed_model.calls == 1

    failed_model = DeterministicAcceptanceModel(fail=True)
    failed_app = create_app(config=_config(database_path), model=failed_model)
    failed_request = str(uuid4())
    with TestClient(failed_app) as client:
        user_id = int(_resolve(client, "failed-replay")["user"]["id"])
        first_failure = _stream(
            client, user_id, "same failed request", request_id=failed_request
        )
        failure_replay = _stream(
            client, user_id, "same failed request", request_id=failed_request
        )

    assert first_failure[-1] == (
        "error",
        {"code": "model_error", "message": "model generation failed"},
    )
    assert failure_replay == [
        (
            "error",
            {"code": "model_error", "message": "model generation failed"},
        )
    ]
    assert failed_model.calls == 1


def test_same_sqlite_path_preserves_completed_turn_and_checkpoint_across_restart(
    tmp_path,
) -> None:
    """Catches app restart recreating identity or losing private facts/checkpoints."""
    database_path = tmp_path / "restart-completed.sqlite3"
    config = _config(database_path)
    request_id = str(uuid4())
    first_model = DeterministicAcceptanceModel()
    first_app = create_app(config=config, model=first_model)

    with TestClient(first_app) as client:
        first_identity = _resolve(client, "restart-completed")
        user_id = int(first_identity["user"]["id"])
        conversation_id = str(first_identity["conversation"]["id"])
        assert _stream(
            client, user_id, "restart body", request_id=request_id
        )[-1][0] == "done"

        async def first_private_state():
            conversation = await first_app.state.conversations.get_default_by_user(
                user_id
            )
            assert conversation is not None
            checkpoint = await first_app.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": conversation.thread_id}}
            )
            return conversation.thread_id, checkpoint

        thread_id, first_checkpoint = client.portal.call(first_private_state)

    assert first_checkpoint is not None

    second_model = DeterministicAcceptanceModel()
    second_app = create_app(config=config, model=second_model)
    with TestClient(second_app) as client:
        second_identity = _resolve(client, "restart-completed")
        history_response = client.get(f"/api/users/{user_id}/messages")

        async def second_private_state():
            conversation = await second_app.state.conversations.get_default_by_user(
                user_id
            )
            assert conversation is not None
            checkpoint = await second_app.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": conversation.thread_id}}
            )
            return conversation.thread_id, checkpoint

        restarted_thread_id, restarted_checkpoint = client.portal.call(
            second_private_state
        )

    assert second_identity == first_identity
    assert history_response.status_code == 200
    assert [message["content"] for message in history_response.json()["messages"]] == [
        "restart body",
        "答复：restart body",
    ]
    assert restarted_thread_id == thread_id
    assert restarted_checkpoint is not None
    assert second_model.calls == 0

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT conversation_id, status, content, reasoning_content, prompt_json,
                   provider, model, parameters_json, input_tokens, output_tokens,
                   total_tokens, latency_ms, finish_reason, trace_json
            FROM messages
            WHERE request_id = ? AND role = 'assistant'
            """,
            (request_id,),
        ).fetchone()
        stored_thread_id = connection.execute(
            "SELECT thread_id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()[0]

    assert row is not None
    (
        stored_conversation_id,
        status,
        content,
        reasoning,
        prompt_json,
        provider,
        model_name,
        parameters_json,
        input_tokens,
        output_tokens,
        total_tokens,
        latency_ms,
        finish_reason,
        trace_json,
    ) = row
    assert stored_conversation_id == conversation_id
    assert stored_thread_id == thread_id
    assert status == "completed"
    assert content == "答复：restart body"
    assert reasoning == "依据：restart body"
    assert json.loads(prompt_json)[-1] == {"content": "restart body", "role": "user"}
    assert provider == "acceptance-offline"
    assert model_name == "acceptance-offline"
    assert json.loads(parameters_json) == {
        "model": "acceptance-offline",
        "temperature": 0,
    }
    assert (input_tokens, output_tokens, total_tokens) == (7, 3, 10)
    assert isinstance(latency_ms, int) and latency_ms >= 0
    assert finish_reason == "stop"
    trace = json.loads(trace_json)
    assert [node["name"] for node in trace["nodes"]] == [
        "prepare_turn",
        "decide_emotion",
        "analyze_emotion",
        "generate_response",
        "finalize_turn",
    ]
    assert trace["model_calls"][0]["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


def test_restart_fails_streaming_turn_and_resets_only_its_checkpoint(tmp_path) -> None:
    """Catches interrupted recovery deleting a healthy user's checkpoint as collateral."""
    database_path = tmp_path / "restart-interrupted.sqlite3"
    config = _config(database_path)
    first_app = create_app(
        config=config, model=DeterministicAcceptanceModel()
    )
    interrupted_request = str(uuid4())

    with TestClient(first_app) as client:
        interrupted = _resolve(client, "restart-interrupted")
        healthy = _resolve(client, "restart-healthy")
        interrupted_user_id = int(interrupted["user"]["id"])
        healthy_user_id = int(healthy["user"]["id"])

        async def seed_interrupted_state():
            interrupted_conversation = (
                await first_app.state.conversations.get_default_by_user(
                    interrupted_user_id
                )
            )
            healthy_conversation = (
                await first_app.state.conversations.get_default_by_user(healthy_user_id)
            )
            assert interrupted_conversation is not None
            assert healthy_conversation is not None
            turn = await first_app.state.messages.reserve_turn(
                interrupted_conversation.id,
                interrupted_request,
                "unfinished body",
            )
            await first_app.state.messages.mark_streaming(turn.assistant.id)
            for thread_id in (
                interrupted_conversation.thread_id,
                healthy_conversation.thread_id,
            ):
                await first_app.state.checkpointer.aput(
                    {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                    empty_checkpoint(),
                    {"source": "input", "step": -1, "parents": {}},
                    {},
                )
            return interrupted_conversation.thread_id, healthy_conversation.thread_id

        interrupted_thread, healthy_thread = client.portal.call(
            seed_interrupted_state
        )

    second_app = create_app(
        config=config, model=DeterministicAcceptanceModel()
    )
    with TestClient(second_app) as client:
        history = client.get(f"/api/users/{interrupted_user_id}/messages")

        async def recovered_checkpoints():
            interrupted_checkpoint = await second_app.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": interrupted_thread}}
            )
            healthy_checkpoint = await second_app.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": healthy_thread}}
            )
            return interrupted_checkpoint, healthy_checkpoint

        interrupted_checkpoint, healthy_checkpoint = client.portal.call(
            recovered_checkpoints
        )
        recovery_report = second_app.state.recovery_report

    assert history.status_code == 200
    assistant = history.json()["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["status"] == "failed"
    assert assistant["error_code"] == "process_interrupted"
    assert assistant["error_message"] == "generation interrupted"
    assert interrupted_checkpoint is None
    assert healthy_checkpoint is not None
    assert recovery_report.failed_messages == 1
    assert recovery_report.reset_threads == 1
