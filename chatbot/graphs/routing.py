"""Pure, deterministic routing decisions for the conversation graph."""

from typing import Literal

from chatbot.graphs.state import ConversationState


def route_operation(state: ConversationState) -> Literal["turn", "regenerate", "onboard", "invalid"]:
    operation = state.get("operation")
    if operation in {"turn", "regenerate", "onboard"}:
        return operation
    return "invalid"


def should_analyze_emotion(state: ConversationState, *, emotion_interval: int) -> bool:
    risk = state.get("risk") or {}
    if risk.get("force_emotion_analysis") is True:
        return True
    turn_count = state.get("turn_count", 0)
    last_turn = state.get("last_emotion_analysis_turn", 0)
    if last_turn == 0:
        return True
    return turn_count - last_turn >= emotion_interval


def route_emotion(
    state: ConversationState,
    *,
    emotion_interval: int,
) -> Literal["analyze_emotion", "reuse_emotion"]:
    if should_analyze_emotion(state, emotion_interval=emotion_interval):
        return "analyze_emotion"
    return "reuse_emotion"


def route_by_safety(state: ConversationState) -> Literal["generate_reply", "generate_crisis_reply"]:
    safety_state = state.get("safety_state") or {}
    if safety_state.get("level") == "crisis":
        return "generate_crisis_reply"
    return "generate_reply"
