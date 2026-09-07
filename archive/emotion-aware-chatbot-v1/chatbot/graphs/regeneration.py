"""Regenerate one existing AI message without creating another conversation turn."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.nodes.context import load_context
from chatbot.graphs.nodes.generation import build_chat_prompt
from chatbot.graphs.requests import regeneration_fingerprint, request_binding_matches
from chatbot.graphs.state import ConversationState
from chatbot.models.graph import GraphContext, SafetyDecision


APPLICATION_AUDIT_FIELDS = (
    "turn_count",
    "emotion_state",
    "predicted_emotion",
    "safety_level",
    "safety_note",
)
REGENERATION_REASONS = {
    "不准确",
    "不完整",
    "没有理解我的问题",
    "语气不合适",
    "其他",
}


class RegenerationError(ValueError):
    """Stable domain failure raised by the regeneration child graph."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def build_regeneration_graph(deps: NodeDependencies) -> StateGraph:
    """Build the uncompiled message-replacement graph with shared dependencies."""
    builder = StateGraph(ConversationState, context_schema=GraphContext)
    builder.add_node("validate_target", validate_target)
    builder.add_node("replay_completed_regeneration", replay_completed_regeneration)
    builder.add_node("rebuild_context", rebuild_context)
    builder.add_node("load_context", partial(load_context, deps=deps))
    builder.add_node("generate_variant", partial(generate_variant, deps=deps))
    builder.add_node(
        "finalize_regeneration",
        partial(finalize_regeneration, deps=deps),
    )
    builder.add_edge(START, "validate_target")
    builder.add_conditional_edges(
        "validate_target",
        _route_after_validation,
        {
            "replay_completed_regeneration": "replay_completed_regeneration",
            "rebuild_context": "rebuild_context",
        },
    )
    builder.add_edge("replay_completed_regeneration", END)
    builder.add_edge("rebuild_context", "load_context")
    builder.add_edge("load_context", "generate_variant")
    builder.add_edge("generate_variant", "finalize_regeneration")
    builder.add_edge("finalize_regeneration", END)
    return builder


def validate_target(state: ConversationState, runtime) -> dict[str, Any]:
    """Validate the target, UI reason, and default single-regeneration limit."""
    request_id = state.get("request_id") or runtime.context["request_id"]
    fingerprint = regeneration_fingerprint(
        state.get("target_message_id", ""), state.get("regeneration_reason", "")
    )
    completed = state.get("processed_requests", {}).get(request_id, {})
    if completed.get("status") == "completed":
        if not request_binding_matches(completed, "regenerate", fingerprint):
            raise RegenerationError("request_id_conflict")
        return {"replay_request": True, "request_fingerprint": fingerprint}
    reason = state.get("regeneration_reason")
    if reason not in REGENERATION_REASONS:
        raise RegenerationError("invalid_reason")
    target = _eligible_target_message(state)
    if target.additional_kwargs.get("regenerated") is True:
        raise RegenerationError("already_regenerated")
    return {"replay_request": False, "request_fingerprint": fingerprint}


def _route_after_validation(
    state: ConversationState,
) -> Literal["replay_completed_regeneration", "rebuild_context"]:
    if state.get("replay_request") is True:
        return "replay_completed_regeneration"
    return "rebuild_context"


def replay_completed_regeneration(
    state: ConversationState,
    runtime,
    writer,
) -> dict[str, Any]:
    """Replay the exact stored done payload for an idempotent request retry."""
    request_id = state.get("request_id") or runtime.context["request_id"]
    completed = state.get("processed_requests", {}).get(request_id, {})
    event_data = completed.get("event_data")
    if not isinstance(event_data, dict):
        raise RegenerationError("completed_request_invalid")
    writer({"event": "done", "data": {**dict(event_data), "replayed": True}})
    return _regeneration_cleanup_delta()


def rebuild_context(state: ConversationState) -> dict[str, str]:
    """Locate the HumanMessage directly associated with the target AI reply."""
    messages = list(state.get("messages", []))
    target_index = _target_index(messages, state.get("target_message_id", ""))
    for message in reversed(messages[:target_index]):
        if isinstance(message, HumanMessage):
            question = _message_text(message)
            if question:
                return {"input_message": question}
            break
    raise RegenerationError("missing_target")


async def generate_variant(
    state: ConversationState,
    runtime,
    writer,
    config: RunnableConfig,
    *,
    deps: NodeDependencies,
) -> dict[str, Any]:
    """Generate a reason-aware replacement through the async observable model call."""
    target, question, history = _regeneration_dialogue(state)
    prompt = build_chat_prompt(
        profile_context=state.get("profile_context", ""),
        memory_context=state.get("memory_context", ""),
        emotion_context=_target_emotion_context(target),
        safety=_target_safety(target),
    )
    reason = state["regeneration_reason"]
    prompt_value = prompt.invoke(
        {
            "messages": [
                *history,
                SystemMessage(
                    content=(
                        "Regeneration Request:\n"
                        f"- reason: {reason}\n"
                        "Create a fresh answer to the original question. "
                        "Address the selected issue without referring to this instruction "
                        "or to the previous answer."
                    )
                ),
            ],
            "input": question,
        }
    )
    try:
        result = await deps.chat_model.ainvoke(prompt_value, config=config)
        content = _complete_content(result)
    except Exception as exc:
        raise RegenerationError("generation_failed") from exc
    target = _eligible_target_message(state)
    original_audit = _original_application_audit(target)
    metadata = {
        **original_audit,
        **_model_additional_kwargs(result),
        "original_content": _message_text(target),
        "original_audit": original_audit,
        "regeneration_reason": state["regeneration_reason"],
        "regenerated_at": _utc_timestamp(deps.now()),
        "regenerated": True,
    }
    replacement = AIMessage(
        id=target.id,
        content=content,
        additional_kwargs=metadata,
        response_metadata=_model_response_metadata(result),
        usage_metadata=_model_usage_metadata(result),
        name=_model_name(result),
    )
    return {
        "messages": [replacement],
        "response_content": content,
        "response_message_id": str(target.id),
    }


def finalize_regeneration(
    state: ConversationState,
    runtime,
    writer,
    *,
    deps: NodeDependencies,
) -> dict[str, Any]:
    """Record the completed request, emit done, and clear operation transients."""
    request_id = state.get("request_id") or runtime.context["request_id"]
    message_id = state["response_message_id"]
    content = state["response_content"]
    reason = state["regeneration_reason"]
    completed_at = _utc_timestamp(deps.now())
    event_data = {
        "message_id": message_id,
        "content": content,
        "reason": reason,
        "regenerated": True,
    }
    processed_requests = {
        **state.get("processed_requests", {}),
        request_id: {
            "status": "completed",
            "response_message_id": message_id,
            "content": content,
            "completed_at": completed_at,
            "event_data": event_data,
            "operation": "regenerate",
            "input_fingerprint": state["request_fingerprint"],
        },
    }
    writer({"event": "done", "data": event_data})
    return {
        "processed_requests": _newest_requests(
            processed_requests,
            current_request_id=request_id,
            limit=deps.graph_config.request_history_limit,
        ),
        "thread_meta": {
            **state.get("thread_meta", {}),
            "updated_at": completed_at,
        },
        **_regeneration_cleanup_delta(),
    }


def _target_message(state: ConversationState) -> BaseMessage:
    messages = list(state.get("messages", []))
    index = _target_index(messages, state.get("target_message_id", ""))
    return messages[index]


def _eligible_target_message(state: ConversationState) -> AIMessage:
    target = _target_message(state)
    if (
        not isinstance(target, AIMessage)
        or target.tool_calls
        or target.invalid_tool_calls
    ):
        raise RegenerationError("non_ai_target")
    return target


def _target_index(messages: list[BaseMessage], target_id: str) -> int:
    for index, message in enumerate(messages):
        if message.id == target_id:
            return index
    raise RegenerationError("missing_target")


def _regeneration_dialogue(
    state: ConversationState,
) -> tuple[AIMessage, str, list[BaseMessage]]:
    messages = list(state.get("messages", []))
    target_index = _target_index(messages, state.get("target_message_id", ""))
    target = messages[target_index]
    if (
        not isinstance(target, AIMessage)
        or target.tool_calls
        or target.invalid_tool_calls
    ):
        raise RegenerationError("non_ai_target")
    for human_index in range(target_index - 1, -1, -1):
        message = messages[human_index]
        if isinstance(message, HumanMessage):
            question = _message_text(message)
            if question:
                return target, question, messages[:human_index]
            break
    raise RegenerationError("missing_target")


def _target_emotion_context(target: AIMessage) -> str:
    emotion_state = target.additional_kwargs.get("emotion_state")
    if not isinstance(emotion_state, dict) or not emotion_state:
        return ""
    lines = [
        f"- {key}: {value}"
        for key, value in emotion_state.items()
        if isinstance(value, (str, int, float, bool))
    ]
    return "\n".join(lines)


def _target_safety(target: AIMessage) -> SafetyDecision:
    level = target.additional_kwargs.get("safety_level")
    if level not in {"normal", "supportive", "crisis"}:
        level = "normal"
    guidance = {
        "normal": "Reply naturally while preserving ordinary safety boundaries.",
        "supportive": "Keep the original supportive safety posture.",
        "crisis": "Keep the original crisis-safety posture and actionable support.",
    }[level]
    return {"level": level, "guidance": guidance}


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _complete_content(result: Any) -> str:
    value = result.content if hasattr(result, "content") else result
    content = value if isinstance(value, str) else str(value)
    content = content.strip()
    if not content:
        raise ValueError("empty model reply")
    return content


def _model_additional_kwargs(result: Any) -> dict[str, Any]:
    value = getattr(result, "additional_kwargs", {})
    return dict(value) if isinstance(value, dict) else {}


def _original_application_audit(target: AIMessage) -> dict[str, Any]:
    return {
        key: target.additional_kwargs[key]
        for key in APPLICATION_AUDIT_FIELDS
        if key in target.additional_kwargs
    }


def _model_response_metadata(result: Any) -> dict[str, Any]:
    value = getattr(result, "response_metadata", {})
    return dict(value) if isinstance(value, dict) else {}


def _model_usage_metadata(result: Any) -> dict[str, Any] | None:
    value = getattr(result, "usage_metadata", None)
    return dict(value) if isinstance(value, dict) else None


def _model_name(result: Any) -> str | None:
    value = getattr(result, "name", None)
    return value if isinstance(value, str) else None


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


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


def _regeneration_cleanup_delta() -> dict[str, Any]:
    return {
        "operation": "",
        "request_id": "",
        "request_fingerprint": "",
        "input_message": "",
        "target_message_id": "",
        "regeneration_reason": "",
        "profile_answers": [],
        "profile_draft": {},
        "risk": {},
        "profile_context": "",
        "memory_context": "",
        "response_content": "",
        "response_message_id": "",
        "error_code": "",
        "memory_warning": "",
        "replay_request": False,
    }
