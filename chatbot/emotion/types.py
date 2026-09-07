from __future__ import annotations
from dataclasses import dataclass, asdict
from math import isfinite
from collections.abc import Sequence
from typing import Protocol
from langchain_core.messages import BaseMessage
from chatbot.llm.types import TokenUsage

@dataclass(frozen=True)
class Taxonomy:
    labels: dict[str, str]
    families: dict[str, str]
    content_hash: str

@dataclass(frozen=True)
class BudgetConfig:
    context_tokens: int
    output_tokens: int
    safety_tokens: int
    history_ratio: float = 0.60

    def __post_init__(self):
        if (any(type(v) is not int or v < 0 for v in
                (self.context_tokens, self.output_tokens, self.safety_tokens))
            or self.context_tokens <= 0 or self.output_tokens <= 0
            or self.output_tokens + self.safety_tokens >= self.context_tokens
            or type(self.history_ratio) not in (int, float)
            or not isfinite(self.history_ratio) or not 0 < self.history_ratio <= 1):
            raise ValueError('invalid emotion budget')

class TokenCounter(Protocol):
    identity: str
    version: str
    def count(self, messages: Sequence[BaseMessage]) -> int: ...

@dataclass(frozen=True)
class RetrievalConfig:
    example_limit: int = 4
    prior_boost: float = 2.0
    recent_label_limit: int = 3

    def __post_init__(self):
        if (any(type(v) is not int or v < 0 for v in (self.example_limit, self.recent_label_limit))
                or type(self.prior_boost) not in (int, float)
                or not isfinite(self.prior_boost) or self.prior_boost < 0):
            raise ValueError('invalid emotion retrieval settings')

@dataclass(frozen=True)
class EmotionResult:
    primary_emotion: str
    confidence: float
    secondary_emotions: list[str]
    evidence: str
    reply_strategy: str
    trajectory_note: str
    safety_level: str
    primary_family: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class ModelOutcome:
    raw_output: str | None
    reasoning_content: str | None
    usage: TokenUsage
    finish_reason: str | None
    metadata: dict[str, object]
    error: dict[str, object] | None

class EmotionModel(Protocol):
    parameters: dict[str, object]
    async def invoke(self, messages: Sequence[BaseMessage]) -> ModelOutcome: ...
