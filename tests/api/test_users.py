from __future__ import annotations

from fastapi.testclient import TestClient


def test_resolve_creates_and_reuses_trimmed_exact_case_user(
    client: TestClient,
) -> None:
    """Catches resolve leaking thread IDs or creating duplicates after trimming."""
    first = client.post("/api/users/resolve", json={"identifier": " Alice "})
    second = client.post("/api/users/resolve", json={"identifier": "Alice"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["user"]["identifier"] == "Alice"
    assert set(first.json()) == {"user", "conversation"}
    assert set(first.json()["user"]) == {"id", "identifier"}
    assert set(first.json()["conversation"]) == {"id", "title"}
    assert "thread_id" not in first.text


def test_resolve_preserves_identifier_case(client: TestClient) -> None:
    """Catches case-folding that merges distinct exact-match identities."""
    upper = client.post("/api/users/resolve", json={"identifier": "Alice"})
    lower = client.post("/api/users/resolve", json={"identifier": "alice"})

    assert upper.status_code == lower.status_code == 200
    assert upper.json()["user"]["id"] != lower.json()["user"]["id"]


def test_invalid_resolve_payloads_have_one_safe_stable_error_shape(
    client: TestClient,
) -> None:
    """Catches framework validation details or raw identifier values leaking outward."""
    payloads = (
        {},
        {"identifier": 7},
        {"identifier": " \n "},
        {"identifier": "x" * 129},
        {"identifier": "bad\x00value"},
    )

    for payload in payloads:
        response = client.post("/api/users/resolve", json=payload)
        assert response.status_code == 400
        assert response.json() == {
            "error": {
                "code": "invalid_identifier_or_message",
                "message": "invalid identifier or message",
            }
        }
        assert "traceback" not in response.text.lower()
        assert "input" not in response.text.lower()
