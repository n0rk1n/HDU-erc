from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from pydantic import Secret, SecretBytes, SecretStr

from typing import Any

JSONValue = Any


REDACTED = "[REDACTED]"
_UNSERIALIZABLE = "[UNSERIALIZABLE]"
_SENSITIVE_KEYS = {
    "apikey",
    "authorization",
    "clientsecret",
    "cookie",
    "accesstoken",
    "password",
    "passwd",
    "privatekey",
    "refreshtoken",
    "setcookie",
    "secret",
    "xapikey",
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
            normalized_key = _normalize_key(key)
            if normalized_key in _SENSITIVE_KEYS:
                redacted[key] = REDACTED
            elif _is_url_key(normalized_key) and isinstance(item, str):
                redacted[key] = _redact_url(item)
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return _UNSERIALIZABLE


def _normalize_key(key: str) -> str:
    return key.lower().replace("_", "").replace("-", "")


def _is_url_key(normalized_key: str) -> bool:
    return normalized_key.endswith("url") or normalized_key.endswith("uri")


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if parsed.scheme and hostname is None:
            return REDACTED if _looks_authenticated(value) else value

        netloc = parsed.netloc
        if parsed.username is not None or parsed.password is not None:
            if hostname is None:
                return REDACTED
            host = f"[{hostname}]" if ":" in hostname else hostname
            netloc = host if parsed.port is None else f"{host}:{parsed.port}"

        query = urlencode(
            [
                (key, REDACTED if _normalize_key(key) in _SENSITIVE_KEYS else item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            ],
            doseq=True,
            quote_via=quote,
            safe="[]",
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    except (TypeError, ValueError):
        return REDACTED if _looks_authenticated(value) else value


def _looks_authenticated(value: str) -> bool:
    authority = value.partition("://")[2].partition("/")[0]
    if "@" in authority or ":" in authority:
        return True
    query = value.partition("?")[2].partition("#")[0]
    return any(
        _normalize_key(partition.partition("=")[0]) in _SENSITIVE_KEYS
        for partition in query.split("&")
        if partition
    )
