from __future__ import annotations

from pydantic import SecretBytes, SecretStr

from chatbot.llm.redaction import redact_secrets


def test_redaction_removes_sensitive_keys_recursively() -> None:
    value = {
        "Authorization": "Bearer secret",
        "nested": {
            "API-KEY": "secret",
            "access_token": "secret",
            "temperature": 0.7,
        },
        "items": [
            {"Set_Cookie": "session=secret", "model": "example"},
            {"secret": "secret"},
        ],
        "cookie": "session=secret",
    }

    assert redact_secrets(value) == {
        "Authorization": "[REDACTED]",
        "nested": {
            "API-KEY": "[REDACTED]",
            "access_token": "[REDACTED]",
            "temperature": 0.7,
        },
        "items": [
            {"Set_Cookie": "[REDACTED]", "model": "example"},
            {"secret": "[REDACTED]"},
        ],
        "cookie": "[REDACTED]",
    }


def test_redaction_preserves_unrelated_similar_keys() -> None:
    value = {
        "secretary": "Ada",
        "cookie_policy": "strict",
        "authorization_mode": "oauth",
        "public_api_key_hint": "last-four",
    }

    assert redact_secrets(value) == value


def test_redaction_never_serializes_secret_value_objects() -> None:
    assert redact_secrets(
        {
            "password": SecretStr("plain-secret"),
            "certificate": SecretBytes(b"private-bytes"),
        }
    ) == {
        "password": "[REDACTED]",
        "certificate": "[REDACTED]",
    }


def test_redaction_replaces_only_mapping_with_non_string_key() -> None:
    value = {
        "safe": "kept",
        "nested": {1: "must-not-leak", "also": "discarded-with-mapping"},
    }

    assert redact_secrets(value) == {
        "safe": "kept",
        "nested": "[UNSERIALIZABLE]",
    }
