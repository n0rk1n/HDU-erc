import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chatbot.config import ChatConfig, LlmConfig
from chatbot.llm import build_chain, build_llm, build_system_message, format_emotion_context

pytestmark = pytest.mark.filterwarnings("ignore:RunnableWithMessageHistory is deprecated.*")

EXPECTED_CHATBOT_SYSTEM_MESSAGE = (
    "You are an emotionally aware chatbot companion who talks like a real person "
    "in a private chat.\n\n"
    "Reply as if you are texting the user directly. Be warm, simple, and natural. "
    "If one sentence is enough, say one sentence. Most replies should be a short "
    "paragraph, not a structured answer.\n\n"
    "Do not format ordinary chat as Markdown. Avoid headings, bullet lists, "
    "numbered lists, tables, and code blocks unless the user clearly asks for "
    "structure, code, steps, or a comparison.\n\n"
    "Match the user's language and emotional tone. When the user shares feelings, "
    "respond to the feeling first in plain words, then continue naturally. Ask at "
    "most one easy follow-up question. Do not overpromise, diagnose the user, or "
    "pretend to replace professional help."
)


def test_format_emotion_context_empty():
    assert format_emotion_context("") == ""


def test_format_emotion_context_with_label():
    assert format_emotion_context("anxious") == "Current detected user emotion: anxious"


def test_build_system_message_includes_dynamic_emotion_placeholder():
    message = build_system_message("- name: Alice")

    assert EXPECTED_CHATBOT_SYSTEM_MESSAGE in message
    assert "User Profile:\n- name: Alice" in message
    assert "{emotion_context}" in message


def test_build_chain_injects_emotion_context_into_system_message():
    captured_messages = []

    def fake_llm(prompt_value):
        captured_messages.extend(prompt_value.to_messages())
        return AIMessage(content="ok")

    chain = build_chain(RunnableLambda(fake_llm), "- name: Alice")

    result = chain.invoke(
        {
            "input": "hello",
            "emotion_context": "Current detected user emotion: anxious",
        },
        config={"configurable": {"session_id": "test-emotion-context"}},
    )

    assert result.content == "ok"
    assert captured_messages[0].content == (
        f"{EXPECTED_CHATBOT_SYSTEM_MESSAGE}\n\n"
        "User Profile:\n- name: Alice\n\n"
        "Current detected user emotion: anxious"
    )


def test_build_chain_defaults_missing_emotion_context_to_empty_string():
    captured_messages = []

    def fake_llm(prompt_value):
        captured_messages.extend(prompt_value.to_messages())
        return AIMessage(content="ok")

    chain = build_chain(RunnableLambda(fake_llm), "- name: Alice")

    result = chain.invoke(
        {"input": "hello"},
        config={"configurable": {"session_id": "test-missing-emotion-context"}},
    )

    assert result.content == "ok"
    assert captured_messages[0].content == (
        f"{EXPECTED_CHATBOT_SYSTEM_MESSAGE}\n\n"
        "User Profile:\n- name: Alice\n\n"
    )


def test_build_llm_accepts_chat_config_and_uses_chat_llm(monkeypatch):
    captured_configs = []

    def fake_build_chat_model(config):
        captured_configs.append(config)
        return "fake-model"

    monkeypatch.setattr("chatbot.llm.build_chat_model", fake_build_chat_model)
    chat_llm = LlmConfig(
        provider="deepseek",
        api_key="chat-key",
        model="deepseek-chat",
        temperature=0.2,
        base_url="https://api.deepseek.com/v1",
    )
    emotion_llm = LlmConfig(
        provider="openai",
        api_key="emotion-key",
        model="gpt-4o-mini",
        temperature=0.1,
    )
    config = ChatConfig(chat_llm=chat_llm, emotion_llm=emotion_llm)

    llm = build_llm(config)

    assert llm == "fake-model"
    assert captured_configs == [chat_llm]
