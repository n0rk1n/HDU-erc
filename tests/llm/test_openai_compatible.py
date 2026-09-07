from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import SecretStr

from chatbot.core.config import AppConfig
from chatbot.db.models import Message
from chatbot.llm.openai_compatible import (
    ModelStreamError,
    OpenAICompatibleChatModel,
    parse_ai_message_chunk,
)
from chatbot.llm.prompt import SYSTEM_PROMPT, build_prompt


def _config() -> AppConfig:
    return AppConfig(
        llm_api_key=SecretStr("sk-unit-secret"),
        llm_model="unit-model",
        llm_base_url="https://example.invalid/v1",
        llm_temperature=0.25,
        llm_timeout_seconds=17.0,
        context_message_limit=40,
        sqlite_db_path=Path("unused.sqlite3"),
    )


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
@pytest.mark.parametrize("base_url", ["https://api.deepseek.com", "https://api.deepseek.com/v1"])
@pytest.mark.parametrize("thinking", ["disabled", "enabled"])
def test_deepseek_chat_requests_use_configured_thinking(model, base_url, thinking) -> None:
    """Catches reply requests or audit ignoring the configured thinking mode."""
    adapter = OpenAICompatibleChatModel(
        replace(_config(), llm_model=model, llm_base_url=base_url, llm_thinking=thinking)
    )
    payload = adapter.client._get_request_payload([HumanMessage(content="你好")])
    assert payload["extra_body"] == {"thinking": {"type": thinking}}
    assert adapter.parameters["extra_body"] == {"thinking": {"type": thinking}}


@pytest.mark.parametrize("model,base_url", [
    ("gpt-4o-mini", None),
    ("deepseek-v4-flash", "https://example.invalid/v1"),
    ("deepseek-v4-flash", "https://api.deepseek.com.example.invalid/v1"),
])
def test_other_endpoints_do_not_receive_deepseek_thinking_fields(model, base_url) -> None:
    """Catches a provider-specific option leaking into unrelated API contracts."""
    adapter = OpenAICompatibleChatModel(
        replace(_config(), llm_model=model, llm_base_url=base_url)
    )
    payload = adapter.client._get_request_payload([HumanMessage(content="你好")])
    assert not payload.get("extra_body")
    assert "extra_body" not in adapter.parameters


@pytest.mark.parametrize('base_url', [
    'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
])
def test_glm53_rejects_disabled_thinking(base_url):
    from chatbot.core.errors import ConfigError
    with pytest.raises(ConfigError, match='GLM-5.3.*enabled'):
        OpenAICompatibleChatModel(replace(
            _config(), llm_model='ZHIPU/GLM-5.3', llm_base_url=base_url,
            llm_thinking='disabled'))


def _message(*, role: str, content: str, sequence_no: int) -> Message:
    return Message(
        id=f"message-{sequence_no}",
        conversation_id="conversation-1",
        request_id=f"request-{sequence_no}",
        sequence_no=sequence_no,
        role=role,
        status="completed",
        content=content,
        reasoning_content=None,
        trace_json='{"schema_version":1,"events":[]}',
        prompt_json=None,
        provider=None,
        model=None,
        parameters_json=None,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        latency_ms=None,
        finish_reason=None,
        error_code=None,
        error_message=None,
        created_at="2026-09-03T00:00:00Z",
        updated_at="2026-09-03T00:00:00Z",
        completed_at="2026-09-03T00:00:00Z",
    )


def test_build_prompt_accepts_repository_context_and_preserves_order() -> None:
    context = [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一个回答"},
        {"role": "user", "content": "第二个问题"},
    ]

    prompt = build_prompt(context)

    assert [type(message) for message in prompt] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [message.content for message in prompt] == [
        SYSTEM_PROMPT,
        "第一个问题",
        "第一个回答",
        "第二个问题",
    ]
    from chatbot.llm.prompt import get_system_prompt
    assert prompt[0].content == get_system_prompt()


def test_build_prompt_also_accepts_persisted_message_objects() -> None:
    prompt = build_prompt(
        [
            _message(role="user", content="问题", sequence_no=1),
            _message(role="assistant", content="回答", sequence_no=2),
        ]
    )

    assert [type(message) for message in prompt] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
    ]
    assert [message.content for message in prompt[1:]] == ["问题", "回答"]


def test_parse_chunk_keeps_actual_reasoning_separate() -> None:
    chunk = AIMessageChunk(
        content="答案",
        additional_kwargs={"reasoning_content": "依据"},
        response_metadata={"finish_reason": "stop"},
        usage_metadata={
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        },
    )

    delta = parse_ai_message_chunk(chunk)

    assert delta.content == "答案"
    assert delta.reasoning == "依据"
    assert delta.usage is not None
    assert delta.usage.input_tokens == 3
    assert delta.usage.output_tokens == 2
    assert delta.usage.total_tokens == 5
    assert delta.finish_reason == "stop"


def test_parse_chunk_does_not_invent_reasoning_from_normal_content() -> None:
    delta = parse_ai_message_chunk(AIMessageChunk(content="普通正文"))

    assert delta.content == "普通正文"
    assert delta.reasoning == ""


def test_parse_chunk_extracts_only_standard_text_content_blocks() -> None:
    chunk = AIMessageChunk(
        content=[
            {"type": "text", "text": "正文一"},
            {"type": "image_url", "image_url": {"url": "secret-image"}},
            "正文二",
            {"type": "tool_call", "name": "lookup", "args": {"q": "x"}},
        ]
    )

    delta = parse_ai_message_chunk(chunk)

    assert delta.content == "正文一正文二"
    assert delta.reasoning == ""


def test_parse_chunk_safely_replaces_metadata_mapping_with_non_string_key() -> None:
    chunk = AIMessageChunk(
        content="答案",
        response_metadata={
            "provider": "compatible",
            "nested": {1: "must-not-leak"},
        },
    )

    delta = parse_ai_message_chunk(chunk)

    assert delta.response_metadata == {
        "provider": "compatible",
        "nested": "[UNSERIALIZABLE]",
    }
    serialized = json.dumps(delta.response_metadata)
    assert "must-not-leak" not in serialized


@pytest.mark.parametrize(
    ("response_metadata", "expected"),
    [
        (
            {
                "token_usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 4,
                    "total_tokens": 11,
                }
            },
            (7, 4, 11),
        ),
        (
            {
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 5,
                    "total_tokens": 13,
                }
            },
            (8, 5, 13),
        ),
    ],
)
def test_parse_chunk_accepts_approved_sdk_usage_shapes(
    response_metadata: dict[str, object], expected: tuple[int, int, int]
) -> None:
    delta = parse_ai_message_chunk(
        AIMessageChunk(content="", response_metadata=response_metadata)
    )

    assert delta.usage is not None
    assert (
        delta.usage.input_tokens,
        delta.usage.output_tokens,
        delta.usage.total_tokens,
    ) == expected


def test_adapter_constructs_real_chat_openai_with_exact_config() -> None:
    adapter = OpenAICompatibleChatModel(_config())
    client = adapter.client

    assert client.model_name == "unit-model"
    assert client.openai_api_base == "https://example.invalid/v1"
    assert client.temperature == 0.25
    assert client.request_timeout == 17.0
    assert client.streaming is True
    assert client.openai_api_key == SecretStr("sk-unit-secret")
    assert adapter.parameters == {
        "provider": "openai-compatible",
        "model": "unit-model",
        "base_url": "https://example.invalid/v1",
        "temperature": 0.25,
        "timeout": 17.0,
        "streaming": True,
    }
    assert "sk-unit-secret" not in repr(adapter)
    assert "sk-unit-secret" not in repr(adapter.parameters)


def test_adapter_repr_does_not_expose_credentials_embedded_in_base_url() -> None:
    """Catches diagnostic repr leaking URL userinfo or sensitive query values."""
    config = replace(
        _config(),
        llm_base_url=(
            "https://user:url-password@example.invalid/v1"
            "?api_key=query-secret&region=cn"
        ),
    )

    rendered = repr(OpenAICompatibleChatModel(config, client=_ChunkSource()))

    assert "url-password" not in rendered
    assert "query-secret" not in rendered
    assert "https://example.invalid/v1" in rendered


class _ChunkSource:
    def __init__(
        self,
        chunks: list[AIMessageChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._error = error
        self.received_prompt: Sequence[BaseMessage] | None = None

    async def astream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        self.received_prompt = prompt
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


@pytest.mark.asyncio
async def test_adapter_stream_keeps_content_reasoning_and_final_usage_chunks() -> None:
    chunks = [
        AIMessageChunk(content="答"),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "依据"}),
        AIMessageChunk(
            content="案",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
        ),
    ]
    source = _ChunkSource(chunks)
    adapter = OpenAICompatibleChatModel(_config(), client=source)
    prompt = [HumanMessage(content="问题")]

    deltas = [delta async for delta in adapter.stream(prompt)]

    assert source.received_prompt is prompt
    assert [delta.content for delta in deltas] == ["答", "", "案"]
    assert [delta.reasoning for delta in deltas] == ["", "依据", ""]
    assert deltas[0].usage is None
    assert deltas[-1].usage is not None
    assert deltas[-1].usage.total_tokens == 5
    assert deltas[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_adapter_error_does_not_expose_api_key() -> None:
    adapter = OpenAICompatibleChatModel(
        _config(), client=_ChunkSource(error=RuntimeError("failed sk-unit-secret"))
    )

    with pytest.raises(ModelStreamError) as captured:
        _ = [delta async for delta in adapter.stream([HumanMessage(content="x")])]

    assert "sk-unit-secret" not in str(captured.value)
    assert "sk-unit-secret" not in repr(captured.value)
