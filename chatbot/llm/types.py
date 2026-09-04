from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from langchain_core.messages import BaseMessage


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelDelta:
    content: str = ""
    reasoning: str = ""
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    response_metadata: dict[str, object] = field(default_factory=dict)


class ChatModelAdapter(Protocol):
    def stream(self, prompt: Sequence[BaseMessage]) -> AsyncIterator[ModelDelta]: ...


def optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None
