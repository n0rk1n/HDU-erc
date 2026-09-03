from __future__ import annotations

from collections.abc import Mapping

from pydantic import Secret, SecretBytes, SecretStr

from chatbot.llm.types import JSONValue


REDACTED = "[REDACTED]"
_UNSERIALIZABLE = "[UNSERIALIZABLE]"
_SENSITIVE_KEYS = {
    "apikey",
    "authorization",
    "cookie",
    "setcookie",
    "accesstoken",
    "secret",
}
_SECRET_TYPES = (Secret, SecretStr, SecretBytes)


def redact_secrets(value: object) -> JSONValue:
    """Return a JSON-safe copy with approved credential fields removed."""
    if isinstance(value, _SECRET_TYPES):
        return REDACTED
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            return _UNSERIALIZABLE
        redacted: dict[str, JSONValue] = {}
        for key, item in value.items():
            redacted[key] = (
                REDACTED
                if _normalize_key(key) in _SENSITIVE_KEYS
                else redact_secrets(item)
            )
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return _UNSERIALIZABLE


def _normalize_key(key: str) -> str:
    return key.lower().replace("_", "").replace("-", "")
