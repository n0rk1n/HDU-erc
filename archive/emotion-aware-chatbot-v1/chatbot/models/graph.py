"""Strict-msgpack-safe value contracts for the durable conversation graph."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, NotRequired, TypedDict


GraphOperation = Literal["turn", "regenerate", "onboard"]
SafetyLevel = Literal["normal", "supportive", "crisis"]
RequestStatus = Literal["completed", "failed", "in_progress"]


class EmotionSnapshot(TypedDict, total=False):
    timestamp: str
    turn_count: int
    primary_emotion: str
    confidence: float
    secondary_emotions: list[str]
    evidence: str
    reply_strategy: str
    trajectory_note: str
    safety_level: SafetyLevel


class SafetyDecision(TypedDict):
    level: SafetyLevel
    guidance: str


class RiskAssessment(TypedDict):
    signals: list[str]
    force_emotion_analysis: bool
    explicit_crisis: bool


class CompletionEventData(TypedDict):
    """Strict-msgpack-safe payload persisted for an idempotent ``done`` replay."""

    message_id: str
    content: str
    replayed: NotRequired[bool]
    reason: NotRequired[str]
    regenerated: NotRequired[bool]


class ThreadMetadata(TypedDict, total=False):
    thread_id: str
    title: str
    created_at: str
    updated_at: str


class ProfileAnswer(TypedDict):
    key: str
    answer: str


class PendingTurn(TypedDict):
    request_id: str
    content: str
    input_fingerprint: str


class RequestResult(TypedDict, total=False):
    status: RequestStatus
    response_message_id: str
    content: str
    error_code: str
    completed_at: str
    event_data: CompletionEventData
    operation: GraphOperation
    input_fingerprint: str


class ConversationInput(TypedDict, total=False):
    operation: GraphOperation
    request_id: str
    input_message: str
    target_message_id: str
    regeneration_reason: str
    profile_answers: list[ProfileAnswer]


class GraphContext(TypedDict):
    client_id: str
    request_id: str
    locale: Literal["zh-CN"]


@dataclass(frozen=True)
class ThreadRecord:
    """Thread metadata persisted with UTC ISO-8601 timestamp strings only."""

    thread_id: str
    title: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _validate_utc_iso8601(self.created_at)
        _validate_utc_iso8601(self.updated_at)


def _validate_utc_iso8601(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ThreadRecord timestamps must be UTC ISO-8601 strings.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("ThreadRecord timestamps must be UTC ISO-8601 strings.")
