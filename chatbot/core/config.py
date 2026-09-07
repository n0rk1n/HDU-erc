from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

from chatbot.core.errors import ConfigError
from chatbot.core.paths import PROJECT_ROOT


@dataclass(frozen=True)
class AppConfig:
    llm_api_key: SecretStr
    llm_model: str
    llm_base_url: str | None
    llm_temperature: float
    llm_timeout_seconds: float
    context_message_limit: int
    sqlite_db_path: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        api_key = _required("LLM_API_KEY")
        model = _optional("LLM_MODEL") or "gpt-4o-mini"
        base_url = _optional("LLM_BASE_URL")
        temperature = _float("LLM_TEMPERATURE", 0.7)
        timeout = _float("LLM_TIMEOUT_SECONDS", 60.0)
        context_limit = _integer("CHAT_CONTEXT_MESSAGE_LIMIT", 40)
        sqlite_path = Path(_optional("SQLITE_DB_PATH") or "data/chatbot.sqlite3")

        if not 0 <= temperature <= 2:
            raise ConfigError("LLM_TEMPERATURE must be between 0 and 2")
        if timeout <= 0:
            raise ConfigError("LLM_TIMEOUT_SECONDS must be positive")
        if not 1 <= context_limit <= 200:
            raise ConfigError("CHAT_CONTEXT_MESSAGE_LIMIT must be between 1 and 200")

        return cls(
            llm_api_key=SecretStr(api_key),
            llm_model=model,
            llm_base_url=base_url,
            llm_temperature=temperature,
            llm_timeout_seconds=timeout,
            context_message_limit=context_limit,
            sqlite_db_path=sqlite_path,
        )


def _optional(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _required(name: str) -> str:
    value = _optional(name)
    if value is None:
        raise ConfigError(f"{name} is required")
    return value


def _float(name: str, default: float) -> float:
    value = _optional(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _integer(name: str, default: int) -> int:
    value = _optional(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
