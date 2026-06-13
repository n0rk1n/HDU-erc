"""Long-term memory interfaces and formatting helpers."""

import os
from dataclasses import dataclass
from typing import Protocol


DEFAULT_MEMORY_DB_PATH = "data/records/memory.sqlite3"
DEFAULT_MEMORY_MAX_RESULTS = 5
MEMORY_CATEGORIES = {"preference", "profile", "goal", "boundary"}


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    category: str
    source: str
    confidence: float
    created_at: str
    updated_at: str
    last_used_at: str | None
    use_count: int


@dataclass(frozen=True)
class MemoryCandidate:
    content: str
    category: str
    source: str = "chat"
    confidence: float = 0.8


@dataclass(frozen=True)
class MemoryRuntimeConfig:
    enabled: bool
    db_path: str
    max_results: int


class MemoryProvider(Protocol):
    def search(self, query: str, *, limit: int) -> list[Memory]:
        raise NotImplementedError

    def remember(self, candidates: list[MemoryCandidate]) -> list[Memory]:
        raise NotImplementedError


class DisabledMemoryProvider:
    def search(self, query: str, *, limit: int) -> list[Memory]:
        return []

    def remember(self, candidates: list[MemoryCandidate]) -> list[Memory]:
        return []


def format_memory_context(memories: list[Memory]) -> str:
    if not memories:
        return ""
    lines = ["Relevant Long-term Memory:"]
    for memory in memories:
        content = memory.content.strip()
        if content:
            lines.append(f"- {content}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def load_memory_config() -> MemoryRuntimeConfig:
    enabled = _parse_bool(os.getenv("MEMORY_ENABLED"), default=True)
    db_path = _clean(os.getenv("MEMORY_DB_PATH")) or DEFAULT_MEMORY_DB_PATH
    max_results = _parse_positive_int(
        os.getenv("MEMORY_MAX_RESULTS"),
        default=DEFAULT_MEMORY_MAX_RESULTS,
    )
    return MemoryRuntimeConfig(
        enabled=enabled,
        db_path=db_path,
        max_results=max_results,
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_bool(value: str | None, *, default: bool) -> bool:
    value = _clean(value)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _parse_positive_int(value: str | None, *, default: int) -> int:
    value = _clean(value)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed
