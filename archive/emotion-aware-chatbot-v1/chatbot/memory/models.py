"""Long-term memory interfaces and formatting helpers."""

import os
from dataclasses import dataclass
from typing import Any, Protocol


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
    max_results: int


class AsyncMemoryRepository(Protocol):
    """Async, client-scoped memory contract for graph runtime nodes."""

    async def asearch(self, client_id: str, query: str, *, limit: int) -> list[Memory]:
        raise NotImplementedError

    async def aremember(
        self, client_id: str, candidates: list[MemoryCandidate]
    ) -> list[Memory]:
        raise NotImplementedError

    async def aget_consolidation_state(self, client_id: str) -> dict[str, Any]:
        raise NotImplementedError

    async def amark_consolidated(
        self,
        client_id: str,
        *,
        turn_count: int,
        last_message_id: str | None,
        source_checkpoint_id: str | None,
    ) -> None:
        raise NotImplementedError


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
    max_results = _parse_positive_int(
        os.getenv("MEMORY_MAX_RESULTS"),
        default=DEFAULT_MEMORY_MAX_RESULTS,
    )
    return MemoryRuntimeConfig(
        enabled=enabled,
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
