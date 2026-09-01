"""Pure asynchronous emotion analysis and deterministic safety nodes."""

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from chatbot.emotion import analyze_emotion_async, assess_safety
from chatbot.emotion.state import EmotionState
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.state import ConversationState, capped_timeline


async def analyze_emotion_node(
    state: ConversationState,
    runtime,
    writer,
    *,
    deps: NodeDependencies,
) -> dict[str, Any]:
    """Analyze the current turn without writing outside durable graph state."""
    writer({"event": "emotion_start", "data": {}})
    records, current_input = _analysis_inputs(state)
    previous = _emotion_from_state(state.get("emotion_state"))
    result = await analyze_emotion_async(
        deps.emotion_model,
        records,
        current_input,
        previous_emotion=previous.primary_emotion if previous else "",
        likely_emotions=state.get("recent_emotions", []),
        turn_count=state.get("turn_count", 0),
        emotion_interval=deps.chat_config.emotion_interval,
    )
    if not result.success or result.state is None:
        writer(
            {
                "event": "emotion_error",
                "data": {"error_code": "emotion_analysis_failed"},
            }
        )
        return {
            "emotion_state": state.get("emotion_state"),
            "error_code": "emotion_analysis_failed",
        }

    emotion_state = result.state.to_dict()
    turn_count = state.get("turn_count", 0)
    snapshot = {
        "timestamp": deps.now().isoformat(),
        "turn_count": turn_count,
        **emotion_state,
    }
    recent_emotions = _remember_recent_emotion(
        state.get("recent_emotions", []), result.state.primary_emotion
    )
    writer(
        {
            "event": "emotion_done",
            "data": {"emotion": result.state.primary_emotion, "state": emotion_state},
        }
    )
    return {
        "emotion_state": emotion_state,
        "emotion_timeline": capped_timeline(
            state.get("emotion_timeline", []),
            [snapshot],
            limit=deps.graph_config.timeline_limit,
        ),
        "recent_emotions": recent_emotions,
        "last_emotion_analysis_turn": turn_count,
        "error_code": "",
    }


def reuse_emotion(state: ConversationState, runtime, writer) -> dict[str, Any]:
    """Reuse checkpointed emotion state without creating redundant writes or events."""
    if "emotion_state" in state:
        return {}
    return {"emotion_state": None}


def assess_safety_node(state: ConversationState, runtime, writer) -> dict[str, Any]:
    """Apply local safety policy after either emotion branch."""
    decision = assess_safety(
        state.get("input_message", ""),
        _emotion_from_state(state.get("emotion_state")),
        risk=state.get("risk"),
    )
    writer({"event": "safety", "data": decision})
    return {"safety_state": decision}


def _analysis_inputs(state: ConversationState) -> tuple[list[dict[str, str]], str]:
    messages = list(state.get("messages", []))
    last_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None,
    )
    if last_human_index is None:
        return _message_records(messages), str(state.get("input_message", "")).strip()
    current_input = _text_content(messages[last_human_index])
    return _message_records(messages[:last_human_index]), current_input


def _message_records(messages: list[BaseMessage]) -> list[dict[str, str]]:
    records = []
    for message in messages:
        content = _text_content(message)
        if content:
            records.append({"role": message.type, "content": content})
    return records


def _text_content(message: BaseMessage) -> str:
    content = message.content
    return content.strip() if isinstance(content, str) else str(content).strip()


def _emotion_from_state(value: dict[str, Any] | None) -> EmotionState | None:
    if not isinstance(value, dict):
        return None
    return EmotionState.from_mapping(value)


def _remember_recent_emotion(current: list[str], emotion: str) -> list[str]:
    return [emotion, *(item for item in current if item != emotion)][:3]
