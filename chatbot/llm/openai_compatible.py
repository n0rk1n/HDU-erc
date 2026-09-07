from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol, cast
from urllib.parse import urlsplit

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI

from chatbot.core.config import AppConfig
from chatbot.llm.redaction import redact_secrets
from chatbot.llm.types import ModelDelta, TokenUsage, optional_string


class ModelStreamError(RuntimeError):
    """A safe model boundary error that contains no provider secrets."""


class _StreamingClient(Protocol):
    def astream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]: ...


class OpenAICompatibleChatModel:
    provider = "openai-compatible"

    def __init__(
        self, config: AppConfig, *, client: _StreamingClient | None = None
    ) -> None:
        self._model = config.llm_model
        self._base_url = config.llm_base_url
        self._temperature = config.llm_temperature
        self._timeout = config.llm_timeout_seconds
        self._extra_body = thinking_extra_body(
            config.llm_model, config.llm_base_url, config.llm_thinking
        )
        self._client: _StreamingClient = (
            client
            if client is not None
            else ChatOpenAI(
                api_key=config.llm_api_key,
                model=config.llm_model,
                base_url=config.llm_base_url,
                temperature=config.llm_temperature,
                timeout=config.llm_timeout_seconds,
                streaming=True,
                extra_body=self._extra_body,
            )
        )

    @property
    def client(self) -> ChatOpenAI:
        """Expose the configured SDK client for construction diagnostics."""
        if not isinstance(self._client, ChatOpenAI):
            raise TypeError("the injected stream client is not ChatOpenAI")
        return self._client

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self._model,
            "base_url": _safe_base_url(self._base_url),
            "temperature": self._temperature,
            "timeout": self._timeout,
            "streaming": True,
            **({"extra_body": self._extra_body} if self._extra_body else {}),
        }

    async def stream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[ModelDelta]:
        try:
            async for chunk in self._client.astream(prompt):
                yield parse_ai_message_chunk(chunk)
        except Exception:
            raise ModelStreamError("OpenAI-compatible model stream failed") from None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self._model!r}, "
            f"base_url={_safe_base_url(self._base_url)!r}, temperature={self._temperature!r}, "
            f"timeout={self._timeout!r}, streaming=True)"
        )


def thinking_extra_body(model: str, base_url: str | None, thinking: str) -> dict | None:
    """Share the official DeepSeek V4 thinking contract across all model roles."""
    if (urlsplit(base_url or "").hostname == "api.deepseek.com"
            and model in {"deepseek-v4-flash", "deepseek-v4-pro"}):
        return {"thinking": {"type": thinking}}
    return None


def parse_ai_message_chunk(chunk: AIMessageChunk) -> ModelDelta:
    metadata_value = redact_secrets(chunk.response_metadata)
    metadata = cast(dict[str, object], metadata_value)
    return ModelDelta(
        content=_text_content(chunk.content),
        reasoning=_reasoning_content(chunk.additional_kwargs),
        usage=_token_usage(chunk.usage_metadata, chunk.response_metadata),
        finish_reason=optional_string(chunk.response_metadata.get("finish_reason")),
        response_metadata=metadata,
    )


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif (
            isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            parts.append(cast(str, block["text"]))
    return "".join(parts)


def _reasoning_content(additional_kwargs: Mapping[str, object]) -> str:
    reasoning = additional_kwargs.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


def _token_usage(
    usage_metadata: Mapping[str, object] | None,
    response_metadata: Mapping[str, object],
) -> TokenUsage | None:
    if usage_metadata is not None:
        return _usage_from_mapping(
            usage_metadata,
            input_key="input_tokens",
            output_key="output_tokens",
        )

    token_usage = response_metadata.get("token_usage")
    if isinstance(token_usage, Mapping):
        return _usage_from_mapping(
            token_usage,
            input_key="prompt_tokens",
            output_key="completion_tokens",
        )

    usage = response_metadata.get("usage")
    if isinstance(usage, Mapping):
        return _usage_from_mapping(
            usage,
            input_key="input_tokens",
            output_key="output_tokens",
        )
    return None


def _usage_from_mapping(
    usage: Mapping[str, object], *, input_key: str, output_key: str
) -> TokenUsage | None:
    input_tokens = _optional_int(usage.get(input_key))
    output_tokens = _optional_int(usage.get(output_key))
    total_tokens = _optional_int(usage.get("total_tokens"))
    if input_tokens is output_tokens is total_tokens is None:
        return None
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _safe_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = redact_secrets({"base_url": value})
    if not isinstance(redacted, dict):
        return "[REDACTED]"
    safe = redacted.get("base_url")
    return safe if isinstance(safe, str) else "[REDACTED]"
