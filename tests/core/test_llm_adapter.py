import pytest
from langchain_core.messages import AIMessage

from chatbot.core.config import LlmConfig
from chatbot.core.llm import build_chain
from chatbot.core.llm_adapter import OpenAICompatibleChatAdapter, build_chat_model

pytestmark = pytest.mark.filterwarnings("ignore:RunnableWithMessageHistory is deprecated.*")


class FakeChatOpenAI:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeChatOpenAI.calls.append(kwargs)

    def invoke(self, prompt, *args, **kwargs):
        return AIMessage(content=f"handled: {prompt}")


def test_build_chat_model_supports_openai_compatible_provider(monkeypatch):
    monkeypatch.setattr("chatbot.core.llm_adapter.ChatOpenAI", FakeChatOpenAI)
    config = LlmConfig(
        provider="deepseek",
        api_key="test-key",
        model="deepseek-chat",
        temperature=0.2,
        base_url="https://api.deepseek.com/v1",
    )

    adapter = build_chat_model(config)

    assert isinstance(adapter, OpenAICompatibleChatAdapter)
    assert FakeChatOpenAI.calls[-1] == {
        "api_key": "test-key",
        "model": "deepseek-chat",
        "temperature": 0.2,
        "base_url": "https://api.deepseek.com/v1",
    }


def test_build_chat_model_omits_empty_base_url(monkeypatch):
    monkeypatch.setattr("chatbot.core.llm_adapter.ChatOpenAI", FakeChatOpenAI)
    config = LlmConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        temperature=0.7,
        base_url=None,
    )

    build_chat_model(config)

    assert FakeChatOpenAI.calls[-1] == {
        "api_key": "test-key",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
    }


def test_adapter_delegates_invoke_to_client(monkeypatch):
    monkeypatch.setattr("chatbot.core.llm_adapter.ChatOpenAI", FakeChatOpenAI)
    config = LlmConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        temperature=0.7,
    )
    adapter = build_chat_model(config)

    response = adapter.invoke("hello")

    assert response.content == "handled: hello"


@pytest.mark.asyncio
async def test_adapter_delegates_ainvoke_to_client(monkeypatch):
    class AsyncFakeChatOpenAI(FakeChatOpenAI):
        async def ainvoke(self, prompt, *args, **kwargs):
            return AIMessage(content=f"async handled: {prompt}")

    monkeypatch.setattr("chatbot.core.llm_adapter.ChatOpenAI", AsyncFakeChatOpenAI)
    config = LlmConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        temperature=0.7,
    )
    adapter = build_chat_model(config)

    response = await adapter.ainvoke("hello")

    assert response.content == "async handled: hello"


def test_adapter_delegates_stream_to_client(monkeypatch):
    class StreamingFakeChatOpenAI(FakeChatOpenAI):
        def stream(self, prompt, *args, **kwargs):
            yield AIMessage(content="hello")
            yield AIMessage(content=" world")

    monkeypatch.setattr("chatbot.core.llm_adapter.ChatOpenAI", StreamingFakeChatOpenAI)
    config = LlmConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        temperature=0.7,
    )
    adapter = build_chat_model(config)

    chunks = list(adapter.stream("hello"))

    assert [chunk.content for chunk in chunks] == ["hello", " world"]


@pytest.mark.asyncio
async def test_adapter_delegates_astream_to_client(monkeypatch):
    class AsyncStreamingFakeChatOpenAI(FakeChatOpenAI):
        async def astream(self, prompt, *args, **kwargs):
            yield AIMessage(content="async hello")
            yield AIMessage(content=" world")

    monkeypatch.setattr("chatbot.core.llm_adapter.ChatOpenAI", AsyncStreamingFakeChatOpenAI)
    config = LlmConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        temperature=0.7,
    )
    adapter = build_chat_model(config)

    chunks = [chunk async for chunk in adapter.astream("hello")]

    assert [chunk.content for chunk in chunks] == ["async hello", " world"]


def test_build_chat_model_rejects_unknown_provider():
    config = LlmConfig(
        provider="unknown",
        api_key="test-key",
        model="some-model",
        temperature=0.7,
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_chat_model(config)


def test_adapter_can_be_composed_in_langchain_chain(monkeypatch):
    monkeypatch.setattr("chatbot.core.llm_adapter.ChatOpenAI", FakeChatOpenAI)
    config = LlmConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-4o-mini",
        temperature=0.7,
    )
    chain = build_chain(build_chat_model(config), "- name: Alice")

    response = chain.invoke(
        {
            "input": "hello",
            "emotion_context": "Current detected user emotion: calm",
        },
        config={"configurable": {"session_id": "test-adapter-composition"}},
    )

    assert response.content.startswith("handled: ")
