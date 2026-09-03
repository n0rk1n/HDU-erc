"""Turn completion, replay, and transient-state cleanup nodes."""

from datetime import datetime, timezone
from typing import Any

from chatbot.core.config import GraphConfig
from chatbot.graphs.state import ConversationState
from chatbot.graphs.requests import turn_fingerprint


def finalize_turn(
    state: ConversationState,
    runtime,
    writer,
    graph_config: GraphConfig,
) -> dict[str, Any]:
    """Record a complete request, emit done, and clear turn-scoped state."""
    request_id = state.get("request_id") or runtime.context["request_id"]
    message_id = state["response_message_id"]
    content = state["response_content"]
    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fingerprint = state.get("request_fingerprint") or turn_fingerprint(
        str(state.get("input_message", ""))
    )
    processed_requests = {
        **state.get("processed_requests", {}),
        request_id: {
            "status": "completed",
            "response_message_id": message_id,
            "content": content,
            "completed_at": completed_at,
            "operation": "turn",
            "input_fingerprint": fingerprint,
        },
    }
    processed_requests = _newest_requests(
        processed_requests,
        current_request_id=request_id,
        limit=graph_config.request_history_limit,
    )
    writer(
        {
            "event": "done",
            "data": {"message_id": message_id, "content": content},
        }
    )
    return {
        "processed_requests": processed_requests,
        "thread_meta": {**state.get("thread_meta", {}), "updated_at": completed_at},
        **_terminal_cleanup_delta(),
    }


def replay_completed_request(
    state: ConversationState,
    runtime,
    writer,
) -> dict[str, Any]:
    """Emit the already completed full response without invoking external dependencies."""
    request_id = state.get("request_id") or runtime.context["request_id"]
    result = state.get("processed_requests", {}).get(request_id, {})
    if result.get("status") != "completed":
        raise ValueError("completed_request_not_found")
    message_id = result.get("response_message_id") or result.get("message_id")
    content = result.get("content")
    if not isinstance(message_id, str) or not isinstance(content, str):
        raise ValueError("completed_request_invalid")
    writer(
        {
            "event": "done",
            "data": {
                "message_id": message_id,
                "content": content,
                "replayed": True,
            },
        }
    )
    return _terminal_cleanup_delta()


def _terminal_cleanup_delta() -> dict[str, Any]:
    return {
        "operation": "",
        "request_id": "",
        "request_fingerprint": "",
        "input_message": "",
        "target_message_id": "",
        "regeneration_reason": "",
        "profile_answers": [],
        "risk": {},
        "profile_context": "",
        "memory_context": "",
        "safety_state": {},
        "response_content": "",
        "response_message_id": "",
        "error_code": "",
        "memory_warning": "",
        "replay_request": False,
        "pending_turn": None,
    }


def _newest_requests(
    processed_requests: dict[str, dict[str, Any]],
    *,
    current_request_id: str,
    limit: int,
) -> dict[str, dict[str, Any]]:
    current = processed_requests[current_request_id]
    remaining = sorted(
        (
            item
            for item in processed_requests.items()
            if item[0] != current_request_id
        ),
        key=lambda item: (str(item[1].get("completed_at", "")), item[0]),
    )
    remaining_limit = max(0, limit - 1)
    newest_remaining = remaining[-remaining_limit:] if remaining_limit else []
    return dict([*newest_remaining, (current_request_id, current)])
