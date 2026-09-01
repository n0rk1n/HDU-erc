"""Input normalization and request idempotency for the turn graph."""

from langchain_core.messages import HumanMessage

from chatbot.graphs.state import ConversationState
from chatbot.graphs.requests import request_binding_matches, turn_fingerprint


class GraphInputError(ValueError):
    """Raised when a graph invocation contains invalid user input."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def accept_turn(state: ConversationState, runtime, writer) -> dict:
    """Normalize one user turn and return only its durable state delta."""
    request_id = state.get("request_id") or runtime.context["request_id"]
    content = str(state.get("input_message", "")).strip()
    fingerprint = turn_fingerprint(content)
    processed = state.get("processed_requests", {}).get(request_id, {})
    if processed.get("status") == "completed":
        if not request_binding_matches(processed, "turn", fingerprint):
            raise GraphInputError("request_id_conflict")
        return {"replay_request": True, "request_fingerprint": fingerprint}

    if not content:
        raise GraphInputError("empty_message")

    message_id = f"human_{request_id}"
    if any(
        getattr(message, "id", None) == message_id
        for message in state.get("messages", [])
    ):
        return {"replay_request": False, "request_fingerprint": fingerprint}

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
        "request_fingerprint": fingerprint,
    }
