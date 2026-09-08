from __future__ import annotations

from pydantic import SecretBytes, SecretStr

from emotion_lab.llm.redaction import redact_secrets


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


def test_redaction_covers_credential_key_variants_and_sanitizes_urls() -> None:
    """Catches credentials surviving via common key spellings or URL components."""
    value = {
        "PASSWORD": "password-secret",
        "Pass_Wd": "passwd-secret",
        "x-api-key": "x-api-secret",
        "API_KEY": "api-secret",
        "refresh-token": "refresh-secret",
        "Client_Secret": "client-secret",
        "private_key": "private-secret",
        "base_url": (
            "https://alice:password-secret@example.invalid/v1/models"
            "?api-key=query-secret&region=cn&refresh_token=refresh-query-secret"
        ),
        "callback_url": "https://example.invalid/ok?region=cn",
        "token_usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    }

    redacted = redact_secrets(value)

    assert redacted == {
        "PASSWORD": "[REDACTED]",
        "Pass_Wd": "[REDACTED]",
        "x-api-key": "[REDACTED]",
        "API_KEY": "[REDACTED]",
        "refresh-token": "[REDACTED]",
        "Client_Secret": "[REDACTED]",
        "private_key": "[REDACTED]",
        "base_url": (
            "https://example.invalid/v1/models"
            "?api-key=[REDACTED]&region=cn&refresh_token=[REDACTED]"
        ),
        "callback_url": "https://example.invalid/ok?region=cn",
        "token_usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    }


def test_redaction_fails_closed_for_malformed_url_with_apparent_credentials() -> None:
    """Catches malformed authenticated URLs being retained after parsing fails."""
    assert redact_secrets({"base_url": "https://user:secret@[invalid"}) == {
        "base_url": "[REDACTED]"
    }


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
