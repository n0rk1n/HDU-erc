"""Serializable event envelope emitted by LangGraph streaming adapters."""

from typing import Any, Literal, TypedDict


GraphEventName = Literal[
    "run_started",
    "user_message",
    "emotion_start",
    "emotion_done",
    "emotion_error",
    "safety",
    "token",
    "done",
    "error",
]


class GraphEvent(TypedDict):
    event: GraphEventName
    data: dict[str, Any]
