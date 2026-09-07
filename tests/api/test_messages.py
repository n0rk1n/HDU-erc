from __future__ import annotations

import sqlite3
from uuid import uuid4

from fastapi.testclient import TestClient

from tests.api.helpers import parse_sse


def _complete_turn(client: TestClient, user_id: int, content: str = "你好") -> str:
    request_id = str(uuid4())
    response = client.post(
        f"/api/users/{user_id}/messages:stream",
        json={"request_id": request_id, "content": content},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert parse_sse(response.text)[-1][0] == "done"
    return request_id


def test_history_returns_only_approved_fields_in_ascending_order(
    client: TestClient, resolved_user, app_config
) -> None:
    """Catches private audit columns leaking through ordinary history serialization."""
    user_id = resolved_user["user"]["id"]
    _complete_turn(client, user_id, "第一问")
    _complete_turn(client, user_id, "第二问")
    with sqlite3.connect(app_config.sqlite_db_path) as connection:
        connection.execute(
            """
            UPDATE messages
            SET reasoning_content = ?, prompt_json = ?, parameters_json = ?, trace_json = ?,
                status = 'failed', error_code = 'model_error', error_message = ?
            WHERE role = 'assistant'
            """,
            (
                "private reasoning",
                '{"api_key":"sk-private"}',
                '{"secret":"private"}',
                '{"traceback":"private"}',
                "raw provider failure: sk-private",
            ),
        )

    response = client.get(f"/api/users/{user_id}/messages")

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert [message["sequence_no"] for message in messages] == [1, 2, 3, 4]
    approved = {
        "id",
        "request_id",
        "sequence_no",
        "role",
        "status",
        "content",
        "bubbles",
        "error_code",
        "error_message",
        "created_at",
        "updated_at",
        "completed_at",
        "processing",
    }
    assert all(set(message) == approved for message in messages)
    failed = [message for message in messages if message["status"] == "failed"]
    assert all(message["error_message"] == "model generation failed" for message in failed)
    serialized = response.text
    for forbidden in (
        "reasoning_content",
        "prompt_json",
        "parameters_json",
        "trace_json",
        "thread_id",
        "sk-private",
        "raw provider failure",
    ):
        assert forbidden not in serialized


def test_history_paginates_backward_then_displays_ascending(
    client: TestClient, resolved_user
) -> None:
    """Catches pagination applying its limit after ascending sort or using <=."""
    user_id = resolved_user["user"]["id"]
    for index in range(3):
        _complete_turn(client, user_id, f"question-{index}")

    latest = client.get(f"/api/users/{user_id}/messages?limit=2")
    earlier = client.get(
        f"/api/users/{user_id}/messages?limit=2&before_sequence=5"
    )

    assert [item["sequence_no"] for item in latest.json()["messages"]] == [5, 6]
    assert [item["sequence_no"] for item in earlier.json()["messages"]] == [3, 4]


def test_history_uses_internal_positive_user_id_and_safe_validation(
    client: TestClient,
) -> None:
    """Catches identifiers being accepted as ownership keys or query bounds escaping validation."""
    missing = client.get("/api/users/999999/messages")
    assert missing.status_code == 404
    assert missing.json() == {
        "error": {"code": "user_not_found", "message": "user not found"}
    }

    for path in (
        "/api/users/Alice/messages",
        "/api/users/0/messages",
        "/api/users/1/messages?limit=0",
        "/api/users/1/messages?limit=201",
        "/api/users/1/messages?before_sequence=0",
    ):
        response = client.get(path)
        assert response.status_code == 400
        assert response.json()["error"] == {
            "code": "invalid_identifier_or_message",
            "message": "invalid identifier or message",
        }
