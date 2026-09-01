"""Durable, strict-msgpack-safe state for the conversation graph."""

from typing import Any, TypedDict

from langgraph.graph import MessagesState

from chatbot.models.graph import (
    EmotionSnapshot,
    GraphOperation,
    ProfileAnswer,
    PendingTurn,
    RequestResult,
    RiskAssessment,
    SafetyDecision,
    ThreadMetadata,
)


def capped_timeline(
    current: list[EmotionSnapshot],
    updates: list[EmotionSnapshot],
    *,
    limit: int,
) -> list[EmotionSnapshot]:
    """Combine timeline updates and retain only the newest serializable snapshots."""
    if limit <= 0:
        return []
    return [*current, *updates][-limit:]


class ConversationState(MessagesState, total=False):
    turn_count: int
    emotion_state: dict[str, Any] | None
    emotion_timeline: list[EmotionSnapshot]
    recent_emotions: list[str]
    safety_state: SafetyDecision
    last_emotion_analysis_turn: int
    thread_meta: ThreadMetadata
    processed_requests: dict[str, RequestResult]
    pending_turn: PendingTurn | None
    operation: GraphOperation
    request_id: str
    request_fingerprint: str
    replay_request: bool
    input_message: str
    target_message_id: str
    regeneration_reason: str
    profile_answers: list[ProfileAnswer]
    profile_draft: dict[str, str]
    risk: RiskAssessment
    profile_context: str
    memory_context: str
    response_content: str
    response_message_id: str
    error_code: str
    memory_warning: str
