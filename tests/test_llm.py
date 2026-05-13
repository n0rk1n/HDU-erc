import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chatbot.llm import build_chain, build_system_message, format_emotion_context

pytestmark = pytest.mark.filterwarnings("ignore:RunnableWithMessageHistory is deprecated.*")


def test_format_emotion_context_empty():
    assert format_emotion_context("") == ""


def test_format_emotion_context_with_label():
    assert format_emotion_context("anxious") == "Current detected user emotion: anxious"


def test_build_system_message_includes_dynamic_emotion_placeholder():
    message = build_system_message("- name: Alice")

    assert "You are a helpful assistant." in message
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
        "You are a helpful assistant.\n\n"
        "User Profile:\n- name: Alice\n\n"
        "Current detected user emotion: anxious"
    )
