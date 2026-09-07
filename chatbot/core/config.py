from __future__ import annotations

import os
from math import isfinite
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
    llm_thinking: str = "disabled"

    def __post_init__(self) -> None:
        if self.llm_thinking not in {"disabled", "enabled"}:
            raise ConfigError("CHAT_LLM_THINKING must be disabled or enabled")

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        model_settings = resolve_model_settings("CHAT_LLM")
        context_limit = _integer("CHAT_CONTEXT_MESSAGE_LIMIT", 40)
        sqlite_path = Path(_optional("SQLITE_DB_PATH") or "data/chatbot.sqlite3")

        if not 1 <= context_limit <= 200:
            raise ConfigError("CHAT_CONTEXT_MESSAGE_LIMIT must be between 1 and 200")

        return cls(
            llm_api_key=model_settings.api_key,
            llm_model=model_settings.model,
            llm_base_url=model_settings.base_url,
            llm_temperature=model_settings.temperature,
            llm_timeout_seconds=model_settings.timeout_seconds,
            context_message_limit=context_limit,
            sqlite_db_path=sqlite_path,
            llm_thinking=_optional("CHAT_LLM_THINKING") or "disabled",
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


@dataclass(frozen=True)
class ModelSettings:
    api_key: SecretStr
    model: str
    base_url: str | None
    temperature: float
    timeout_seconds: float


def resolve_model_settings(prefix: str) -> ModelSettings:
    """Resolve one role directly against LLM_* defaults, never another role."""
    def effective_name(field: str) -> str:
        override = f"{prefix}_{field}"
        return override if _optional(override) is not None else f"LLM_{field}"

    api_key = _required(effective_name("API_KEY"))
    model = _optional(effective_name("MODEL")) or "gpt-4o-mini"
    base_url = _optional(effective_name("BASE_URL"))
    temperature_name = effective_name("TEMPERATURE")
    timeout_name = effective_name("TIMEOUT_SECONDS")
    temperature = _float(temperature_name, 0.7)
    timeout = _float(timeout_name, 60.0)
    if not isfinite(temperature) or not 0 <= temperature <= 2:
        raise ConfigError(f"{temperature_name} must be finite and between 0 and 2")
    if not isfinite(timeout) or timeout <= 0:
        raise ConfigError(f"{timeout_name} must be finite and positive")
    return ModelSettings(SecretStr(api_key), model, base_url, temperature, timeout)
