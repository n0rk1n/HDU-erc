"""Prompt construction and reply generation nodes."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from chatbot.core.prompt_config import load_prompt_config
from chatbot.emotion.state import EmotionState, format_emotion_state_context
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.state import ConversationState
from chatbot.models.graph import SafetyDecision


logger = logging.getLogger(__name__)

CRISIS_FALLBACK_ZH_CN = (
    "我很在意你现在的安全。请先远离可能伤害自己的物品或地方，"
    "立即联系一位你可信任的人陪着你；如果危险迫在眉睫，请拨打当地急救电话"
    "或前往最近的急诊。你不需要一个人扛着。"
)


def build_chat_prompt(
    *,
    profile_context: str,
    memory_context: str,
    emotion_context: str,
    safety: SafetyDecision,
) -> ChatPromptTemplate:
    """Build one prompt while keeping dynamic context in named system sections."""
    sections = [load_prompt_config().chat_system]
    if profile_context.strip():
        sections.append(profile_context.strip())
    if memory_context.strip():
        sections.append(memory_context.strip())
    if emotion_context.strip():
        sections.append(f"Emotion Context:\n{emotion_context.strip()}")
    sections.append(
        "Safety Context:\n"
        f"- level: {safety['level']}\n"
        f"- guidance: {safety['guidance']}"
    )
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="\n\n".join(sections)),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ]
    )


async def generate_reply(
    state: ConversationState,
    runtime,
    writer,
    config: RunnableConfig,
    *,
    deps: NodeDependencies,
) -> dict[str, Any]:
    """Generate one complete normal/supportive reply through the observable model call."""
    prompt = build_chat_prompt(
        profile_context=_profile_context(state.get("profile_context", "")),
        memory_context=state.get("memory_context", ""),
        emotion_context=_emotion_context(state.get("emotion_state")),
        safety=state["safety_state"],
    )
    prompt_value = prompt.invoke(
        {
            "messages": _history_messages(
                state,
                limit=deps.graph_config.request_history_limit,
            ),
            "input": state.get("input_message", ""),
        }
    )
    result = await deps.chat_model.ainvoke(prompt_value, config=config)
    content = _complete_content(result)
    message = _reply_message(state, content)
    return {
        "messages": [message],
        "response_message_id": message.id,
        "response_content": content,
    }


async def generate_crisis_reply(
    state: ConversationState,
    runtime,
    writer,
    config: RunnableConfig,
    *,
    deps: NodeDependencies,
) -> dict[str, Any]:
    """Buffer and validate one crisis reply before exposing any custom event."""
    prompt = build_chat_prompt(
        profile_context=_profile_context(state.get("profile_context", "")),
        memory_context=state.get("memory_context", ""),
        emotion_context=_emotion_context(state.get("emotion_state")),
        safety=state["safety_state"],
    )
    prompt_value = prompt.invoke(
        {
            "messages": _history_messages(
                state,
                limit=deps.graph_config.request_history_limit,
            ),
            "input": state.get("input_message", ""),
        }
    )
    try:
        result = await deps.chat_model.ainvoke(
            prompt_value,
            config=_isolated_crisis_model_config(config),
        )
        content = _complete_content(result)
    except Exception:
        logger.warning(
            "crisis reply generation failed request_id=%s thread_id=%s",
            state.get("request_id", runtime.context.get("request_id", "")),
            _thread_id(state, config),
            exc_info=True,
        )
        content = CRISIS_FALLBACK_ZH_CN
    writer({"event": "token", "data": {"content": content}})
    message = _reply_message(state, content)
    return {
        "messages": [message],
        "response_message_id": message.id,
        "response_content": content,
    }


def _reply_message(state: ConversationState, content: str) -> AIMessage:
    emotion_state = state.get("emotion_state")
    primary_emotion = ""
    if isinstance(emotion_state, dict):
        primary_emotion = str(emotion_state.get("primary_emotion", ""))
    return AIMessage(
        id=f"ai_{state['request_id']}",
        content=content,
        additional_kwargs={
            "feedback": None,
            "turn_count": state["turn_count"],
            "emotion_state": emotion_state,
            "predicted_emotion": primary_emotion,
            "safety_level": state["safety_state"]["level"],
        },
    )


def _complete_content(result: Any) -> str:
    value = result.content if hasattr(result, "content") else result
    content = value if isinstance(value, str) else str(value)
    content = content.strip()
    if not content:
        raise ValueError("chat model returned an empty reply")
    return content


def _profile_context(value: str) -> str:
    value = value.strip()
    if not value or value.startswith("User Profile:"):
        return value
    return f"User Profile:\n{value}"


def _emotion_context(value: dict[str, Any] | None) -> str:
    if not isinstance(value, dict):
        return ""
    emotion = EmotionState.from_mapping(value)
    return format_emotion_state_context(emotion) if emotion is not None else ""


def _history_messages(
    state: ConversationState,
    *,
    limit: int,
) -> list[BaseMessage]:
    messages = [
        message
        for message in state.get("messages", [])
        if isinstance(message, (HumanMessage, AIMessage))
    ]
    current_id = f"human_{state['request_id']}"
    if messages and getattr(messages[-1], "id", None) == current_id:
        messages = messages[:-1]
    return messages[-limit:]


def _thread_id(state: ConversationState, config: RunnableConfig) -> str:
    thread_meta = state.get("thread_meta", {})
    if thread_meta.get("thread_id"):
        return thread_meta["thread_id"]
    configurable = config.get("configurable", {})
    return str(configurable.get("thread_id", ""))


def _isolated_crisis_model_config(config: RunnableConfig) -> RunnableConfig:
    """Preserve request config while preventing pre-validation model stream callbacks."""
    return {**config, "callbacks": []}
