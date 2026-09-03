from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import empty_checkpoint
from pydantic import SecretStr

from chatbot.core.config import AppConfig
from chatbot.core.errors import ConfigError
from chatbot.db.messages import MessageRepository
from chatbot.llm.types import ModelDelta
from chatbot.services.identity import IdentityService
from chatbot.web import create_app


class OfflineModel:
    provider = "offline-test"
    parameters = {"model": "offline-test", "temperature": 0}

    async def stream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[ModelDelta]:
        yield ModelDelta(content="ok")


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    return AppConfig(
        llm_api_key=SecretStr("sk-test-secret"),
        llm_model="offline-test",
        llm_base_url=None,
        llm_temperature=0,
        llm_timeout_seconds=1,
        context_message_limit=20,
        sqlite_db_path=tmp_path / "chatbot.sqlite3",
    )


@pytest.fixture
def offline_model() -> OfflineModel:
    return OfflineModel()


def _pragma(connection, name: str):
    return connection.execute(f"PRAGMA {name}").fetchone()[0]


def test_lifespan_creates_official_saver_tables_and_applies_saver_pragmas(
    app_config, offline_model
) -> None:
    """Catches a separate saver connection missing contention or integrity PRAGMAs."""
    app = create_app(config=app_config, model=offline_model)
    with TestClient(app) as client:
        checkpointer = app.state.checkpointer

        async def read_saver_pragmas():
            values = {}
            for name in ("journal_mode", "foreign_keys", "busy_timeout", "synchronous"):
                row = await (await checkpointer.conn.execute(f"PRAGMA {name}")).fetchone()
                values[name] = row[0]
            return values

        saver_pragmas = client.portal.call(read_saver_pragmas)
        assert app.state.database.path == app_config.sqlite_db_path
        assert app.state.recovery_report.failed_messages == 0

    assert saver_pragmas == {
        "journal_mode": "wal",
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "synchronous": 1,
    }
    with sqlite3.connect(app_config.sqlite_db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"users", "conversations", "messages", "checkpoints", "writes"} <= tables
        assert _pragma(connection, "journal_mode") == "wal"


def test_restart_fails_only_interrupted_business_turn_and_resets_its_checkpoint(
    app_config, offline_model
) -> None:
    """Catches restart deleting all threads or leaving stale running facts active."""
    first_app = create_app(config=app_config, model=offline_model)
    with TestClient(first_app) as client:
        async def seed():
            identity = IdentityService(first_app.state.database)
            normal = await identity.resolve("normal")
            stale = await identity.resolve("stale")
            messages = MessageRepository(first_app.state.database)
            normal_turn = await messages.reserve_turn(
                normal.conversation.id,
                "00000000-0000-4000-8000-000000000001",
                "normal",
            )
            await messages.mark_streaming(normal_turn.assistant.id)
            await messages.complete_assistant(
                normal_turn.assistant.id,
                content="done",
                reasoning_content="private",
                trace={"schema_version": 1},
                prompt=[],
                provider="test",
                model="test",
                parameters={},
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                latency_ms=1,
                finish_reason="stop",
            )
            await messages.reserve_turn(
                stale.conversation.id,
                "00000000-0000-4000-8000-000000000002",
                "stale",
            )
            for conversation in (normal.conversation, stale.conversation):
                await first_app.state.checkpointer.aput(
                    {
                        "configurable": {
                            "thread_id": conversation.thread_id,
                            "checkpoint_ns": "",
                        }
                    },
                    empty_checkpoint(),
                    {"source": "input", "step": -1, "parents": {}},
                    {},
                )
            return normal, stale

        normal, stale = client.portal.call(seed)

    restarted = create_app(config=app_config, model=offline_model)
    with TestClient(restarted) as client:
        async def inspect_restart():
            messages = MessageRepository(restarted.state.database)
            normal_history = await messages.list_visible(normal.conversation.id, limit=10)
            stale_history = await messages.list_visible(stale.conversation.id, limit=10)
            normal_checkpoint = await restarted.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": normal.conversation.thread_id}}
            )
            stale_checkpoint = await restarted.state.checkpointer.aget_tuple(
                {"configurable": {"thread_id": stale.conversation.thread_id}}
            )
            return normal_history, stale_history, normal_checkpoint, stale_checkpoint

        normal_history, stale_history, normal_checkpoint, stale_checkpoint = (
            client.portal.call(inspect_restart)
        )
        report = restarted.state.recovery_report

    assert [message["status"] for message in normal_history] == [
        "completed",
        "completed",
    ]
    assert [message["status"] for message in stale_history] == [
        "completed",
        "failed",
    ]
    assert stale_history[-1]["error_code"] == "process_interrupted"
    assert normal_checkpoint is not None
    assert stale_checkpoint is None
    assert (report.failed_messages, report.reset_threads) == (1, 1)


def test_create_app_without_config_is_import_safe_and_loads_env_at_startup(
    monkeypatch,
) -> None:
    """Catches module import eagerly requiring secrets or returning raw config details."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    app = create_app()

    with pytest.raises(ConfigError, match="LLM_API_KEY is required"):
        with TestClient(app):
            pass

    from chatbot.main import app as module_app

    assert module_app is not None


def test_shutdown_closes_saver_even_when_coordinator_shutdown_fails(
    app_config, offline_model
) -> None:
    """Catches shutdown exceptions leaking the independently owned saver connection."""
    app = create_app(config=app_config, model=offline_model)
    client = TestClient(app)
    client.__enter__()
    events: list[str] = []
    checkpointer = app.state.checkpointer
    original_close = checkpointer.conn.close

    async def failing_shutdown(timeout_seconds: float) -> None:
        events.append("coordinator")
        raise RuntimeError("shutdown failure")

    async def recording_close() -> None:
        events.append("saver")
        await original_close()

    app.state.coordinator.shutdown = failing_shutdown
    checkpointer.conn.close = recording_close

    with pytest.raises(RuntimeError, match="shutdown failure"):
        client.__exit__(None, None, None)

    assert events == ["coordinator", "saver"]


def test_recovery_failure_prevents_half_available_app(
    app_config, offline_model, monkeypatch
) -> None:
    """Catches a failed startup recovery being ignored while routes become available."""
    async def fail_recovery(messages, checkpointer):
        raise RuntimeError("recovery failed")

    monkeypatch.setattr("chatbot.web.recover_interrupted_turns", fail_recovery)
    app = create_app(config=app_config, model=offline_model)

    with pytest.raises(RuntimeError, match="recovery failed"):
        with TestClient(app):
            pass
    assert not hasattr(app.state, "coordinator")
