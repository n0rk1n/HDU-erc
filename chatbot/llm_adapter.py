from typing import Any, Protocol

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from chatbot.config import LlmConfig


class ChatModelAdapter(Protocol):
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError


class OpenAICompatibleChatAdapter(Runnable[Any, Any]):
    def __init__(self, config: LlmConfig):
        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = ChatOpenAI(**kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._client.invoke(input, config=config, **kwargs)


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "deepseek"}


def build_chat_model(config: LlmConfig) -> ChatModelAdapter:
    provider = config.provider.lower()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleChatAdapter(config)
    supported = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS))
    raise ValueError(f"Unsupported LLM provider: {config.provider}. Supported providers: {supported}.")
