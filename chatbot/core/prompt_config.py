"""Shared schema for versioned system prompt configuration files."""

from __future__ import annotations

import json
from pathlib import Path

from chatbot.core.errors import ConfigError


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate prompt config key: {key}")
        result[key] = value
    return result


def load_prompt_config(path: Path) -> dict[str, str]:
    """Read fresh UTF-8 JSON with non-empty version and system strings."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read prompts config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"prompts config {path} must contain a JSON object")
    for key in ("version", "system"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ConfigError(f'prompts config {path} requires a non-empty "{key}" string')
    return {key: data[key] for key in ("version", "system")}
