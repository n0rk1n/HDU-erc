from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from chatbot.core.config import AppConfig
from chatbot.core.errors import ConfigError
from chatbot.web import create_app
from tests.api.helpers import parse_sse


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TEST") != "1",
    reason="set RUN_LIVE_LLM_TEST=1 to call the configured model",
)


@pytest.fixture
def live_config(tmp_path) -> AppConfig:
    try:
        config = AppConfig.from_env()
    except ConfigError as error:
        pytest.skip(f"live model configuration unavailable: {error}")
    return replace(config, sqlite_db_path=tmp_path / "live-smoke.sqlite3")


@pytest.fixture
def live_client(live_config):
    with TestClient(create_app(config=live_config)) as client:
        yield client


def test_live_model_completes_and_persists_trace(live_client, live_config) -> None:
    """Opt-in smoke test for the configured provider and durable audit facts."""
    resolved = live_client.post(
        "/api/users/resolve", json={"identifier": "live-smoke"}
    )
    assert resolved.status_code == 200
    user_id = resolved.json()["user"]["id"]
    request_id = str(uuid4())

    response = live_client.post(
        f"/api/users/{user_id}/messages:stream",
        json={"request_id": request_id, "content": "只回复 OK"},
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1][0] == "done"
    with sqlite3.connect(live_config.sqlite_db_path) as connection:
        row = connection.execute(
            """
            SELECT status, content, reasoning_content, prompt_json, parameters_json,
                   latency_ms, trace_json
            FROM messages
            WHERE request_id = ? AND role = 'assistant'
            """,
            (request_id,),
        ).fetchone()

    assert row is not None
    status, content, reasoning, prompt_json, parameters_json, latency_ms, trace_json = row
    assert status == "completed"
    assert content
    assert reasoning is None or isinstance(reasoning, str)
    assert json.loads(prompt_json)[-1] == {"content": "只回复 OK", "role": "user"}
    assert isinstance(json.loads(parameters_json), dict)
    assert isinstance(latency_ms, int) and latency_ms >= 0
    assert json.loads(trace_json)["model_calls"]
