"""Serializable event envelope emitted by LangGraph streaming adapters."""

from typing import Any, TypedDict


class GraphEvent(TypedDict):
    event: str
    data: dict[str, Any]
