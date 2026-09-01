"""Input normalization and request idempotency for the turn graph."""

from langchain_core.messages import HumanMessage

from chatbot.graphs.state import ConversationState


class GraphInputError(ValueError):
    """Raised when a graph invocation contains invalid user input."""


def accept_turn(state: ConversationState, runtime, writer) -> dict:
    """Normalize one user turn and return only its durable state delta."""
    request_id = state.get("request_id") or runtime.context["request_id"]
    processed = state.get("processed_requests", {}).get(request_id, {})
    if processed.get("status") == "completed":
        return {"replay_request": True}

    content = str(state.get("input_message", "")).strip()
    if not content:
        raise GraphInputError("empty_message")

    message_id = f"human_{request_id}"
    if any(
        getattr(message, "id", None) == message_id
        for message in state.get("messages", [])
    ):
        return {"replay_request": False}

    human = HumanMessage(id=message_id, content=content)
    writer(
        {
            "event": "user_message",
            "data": {"message_id": human.id, "role": "human", "content": human.content},
        }
    )
    return {
        "messages": [human],
        "input_message": content,
        "turn_count": state.get("turn_count", 0) + 1,
        "replay_request": False,
    }
