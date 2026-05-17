"""LLM 适配层 —— 通过 Protocol 抽象不同 provider，当前支持 OpenAI 兼容接口的模型。"""

from typing import Any, Protocol

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from chatbot.config import LlmConfig


class ChatModelAdapter(Protocol):
    """LLM 调用的结构协议，上层仅依赖 invoke/stream，不关心底层 provider 实现。"""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream(self, input: Any, config: Any = None, **kwargs: Any):
        raise NotImplementedError


class OpenAICompatibleChatAdapter(Runnable[Any, Any]):
    """OpenAI 兼容 API 适配器 —— 封装 langchain-openai 的 ChatOpenAI，供 openai/deepseek 等 provider 使用。"""

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

    def stream(self, input: Any, config: Any = None, **kwargs: Any):
        return self._client.stream(input, config=config, **kwargs)


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "deepseek"}


def build_chat_model(config: LlmConfig) -> ChatModelAdapter:
    provider = config.provider.lower()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleChatAdapter(config)
    supported = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS))
    raise ValueError(f"Unsupported LLM provider: {config.provider}. Supported providers: {supported}.")
