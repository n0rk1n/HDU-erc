from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from chatbot.db.models import Message


SYSTEM_PROMPT_VERSION = "v1"
SYSTEM_PROMPT = """系统提示版本：v1
你是一个可靠、清晰且尊重用户的中文助手。请根据对话上下文直接回答；如果信息不足，应明确说明，不要编造事实。"""

ContextMessage = Message | Mapping[str, str]


def build_prompt(messages: Sequence[ContextMessage]) -> list[BaseMessage]:
    """Build the exact model prompt from persisted role/content facts."""
    prompt: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for message in messages:
        role, content = _role_and_content(message)
        if role == "user":
            prompt.append(HumanMessage(content=content))
        elif role == "assistant":
            prompt.append(AIMessage(content=content))
        else:
            raise ValueError(f"unsupported context role: {role}")
    return prompt


def _role_and_content(message: ContextMessage) -> tuple[str, str]:
    if isinstance(message, Message):
        return message.role, message.content
    try:
        role = message["role"]
        content = message["content"]
    except KeyError as exc:
        raise ValueError("context messages require role and content") from exc
    if not isinstance(role, str) or not isinstance(content, str):
        raise TypeError("context role and content must be strings")
    return role, content
