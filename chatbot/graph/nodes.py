from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime

from chatbot.core.errors import InvalidMessageState
from chatbot.db.models import Message, ReservedTurn
from chatbot.graph.dependencies import NodeDependencies
from chatbot.graph.state import TurnContext, TurnState
from chatbot.llm.prompt import build_prompt
from chatbot.llm.redaction import redact_secrets
from chatbot.llm.types import JSONValue, ModelDelta, TokenUsage


@dataclass
class _RunFacts:
    trace: dict[str, object]
    started_at: datetime
    content: str = ""
    reasoning: str = ""
    prompt: list[dict[str, object]] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    parameters: JSONValue = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    response_metadata: dict[str, object] = field(default_factory=dict)
    chunk_count: int = 0
    first_token_ms: int | None = None
    last_flush_at: datetime | None = None
    flushed_characters: int = 0
    publish_failed: bool = False


class TurnNodes:
    """Business-fact-first graph nodes bound to one dependency set."""

    def __init__(self, dependencies: NodeDependencies) -> None:
        self.dependencies = dependencies
        self._runs: dict[str, _RunFacts] = {}

    async def prepare_turn(
        self, state: TurnState, runtime: Runtime[TurnContext]
    ) -> TurnState:
        turn = await self._load_valid_turn(state, runtime.context)
        assistant = turn.assistant
        terminal = _terminal_update(assistant)
        if terminal is not None:
            return terminal
        if assistant.status == "streaming":
            await self.dependencies.messages.fail_assistant(
                assistant.id,
                error_code="process_interrupted",
                error_message="generation interrupted",
            )
            return {"phase": "failed", "error_code": "process_interrupted"}
        if assistant.status != "pending":
            raise InvalidMessageState()

        started_at = self.dependencies.clock()
        facts = _RunFacts(
            trace=_initial_trace(assistant),
            started_at=started_at,
            content=assistant.content,
            reasoning=assistant.reasoning_content or "",
            last_flush_at=started_at,
        )
        await self.dependencies.messages.mark_streaming(assistant.id)
        ended_at = self.dependencies.clock()
        _append_node(facts.trace, "prepare_turn", "completed", started_at, ended_at)
        self._runs[assistant.id] = facts
        return {"phase": "streaming", "error_code": None}

    async def generate_response(
        self, state: TurnState, runtime: Runtime[TurnContext]
    ) -> TurnState:
        turn = await self._load_valid_turn(state, runtime.context)
        assistant = turn.assistant
        terminal = _terminal_update(assistant)
        if terminal is not None:
            return terminal
        if assistant.status != "streaming":
            raise InvalidMessageState()
        facts = self._runs.get(assistant.id)
        if facts is None:
            await self.dependencies.messages.fail_assistant(
                assistant.id,
                error_code="process_interrupted",
                error_message="generation interrupted",
            )
            return {"phase": "failed", "error_code": "process_interrupted"}

        started_at = self.dependencies.clock()
        context_messages = await self.dependencies.messages.list_context(
            runtime.context.conversation_id,
            limit=self.dependencies.context_message_limit,
        )
        prompt = build_prompt(context_messages)
        facts.prompt = [_serialize_message(message) for message in prompt]
        raw_parameters = getattr(self.dependencies.model, "parameters", None)
        facts.parameters = redact_secrets(raw_parameters)
        facts.provider = _optional_string(
            getattr(self.dependencies.model, "provider", None)
        ) or _mapping_string(facts.parameters, "provider")
        facts.model = _mapping_string(facts.parameters, "model")

        error_code: str | None = None
        stream: AsyncIterator[ModelDelta] | None = None
        try:
            stream = self.dependencies.model.stream(prompt).__aiter__()
        except Exception:
            error_code = "model_error"

        while error_code is None and stream is not None:
            try:
                delta = await anext(stream)
            except StopAsyncIteration:
                break
            except Exception:
                error_code = "model_error"
                break

            facts.chunk_count += 1
            facts.content += delta.content
            facts.reasoning += delta.reasoning
            if delta.usage is not None:
                facts.usage = _merge_usage(facts.usage, delta.usage)
            if delta.finish_reason is not None:
                facts.finish_reason = delta.finish_reason
            metadata = redact_secrets(delta.response_metadata)
            if isinstance(metadata, dict):
                facts.response_metadata.update(cast(dict[str, object], metadata))

            now = self.dependencies.clock()
            if facts.first_token_ms is None and delta.content:
                facts.first_token_ms = _duration_ms(started_at, now)
            if delta.content and not facts.publish_failed:
                try:
                    await runtime.context.publisher.publish(
                        "token", {"content": delta.content}
                    )
                except Exception:
                    facts.publish_failed = True
                    _trace_errors(facts.trace).append(
                        {
                            "code": "event_publish_error",
                            "message": "token event publication failed",
                        }
                    )

            if self._should_flush(facts, now):
                _set_stream_trace(facts)
                await self.dependencies.messages.flush_partial(
                    assistant.id,
                    content=facts.content,
                    reasoning_content=facts.reasoning or None,
                    trace=facts.trace,
                )
                facts.last_flush_at = now
                facts.flushed_characters = _generated_characters(facts)

        if error_code is not None:
            facts.finish_reason = "error"
            _trace_errors(facts.trace).append(
                {"code": "model_error", "message": "model generation failed"}
            )

        ended_at = self.dependencies.clock()
        _set_stream_trace(facts)
        _append_model_call(facts)
        _append_node(
            facts.trace,
            "generate_response",
            "failed" if error_code else "completed",
            started_at,
            ended_at,
        )
        await self.dependencies.messages.flush_partial(
            assistant.id,
            content=facts.content,
            reasoning_content=facts.reasoning or None,
            trace=facts.trace,
        )
        facts.last_flush_at = ended_at
        facts.flushed_characters = _generated_characters(facts)
        return {
            "phase": "failed" if error_code else "generated",
            "error_code": error_code,
        }

    async def finalize_turn(
        self, state: TurnState, runtime: Runtime[TurnContext]
    ) -> TurnState:
        turn = await self._load_valid_turn(state, runtime.context)
        assistant = turn.assistant
        terminal = _terminal_update(assistant)
        if terminal is not None:
            return terminal
        if assistant.status != "streaming":
            raise InvalidMessageState()
        facts = self._runs.get(assistant.id)
        if facts is None:
            await self.dependencies.messages.fail_assistant(
                assistant.id,
                error_code="process_interrupted",
                error_message="generation interrupted",
            )
            return {"phase": "failed", "error_code": "process_interrupted"}

        started_at = self.dependencies.clock()
        ended_at = self.dependencies.clock()
        _append_node(facts.trace, "finalize_turn", "completed", started_at, ended_at)
        if state.get("error_code"):
            error_code = cast(str, state["error_code"])
            usage = facts.usage or TokenUsage()
            await self.dependencies.messages.fail_assistant(
                assistant.id,
                error_code=error_code,
                error_message=_safe_error_message(error_code),
                content=facts.content,
                reasoning_content=facts.reasoning or None,
                trace=facts.trace,
                prompt=facts.prompt,
                provider=facts.provider,
                model=facts.model,
                parameters=facts.parameters,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=_duration_ms(facts.started_at, ended_at),
                finish_reason=facts.finish_reason,
            )
            self._runs.pop(assistant.id, None)
            return {"phase": "failed", "error_code": error_code}

        usage = facts.usage or TokenUsage()
        await self.dependencies.messages.complete_assistant(
            assistant.id,
            content=facts.content,
            reasoning_content=facts.reasoning or None,
            trace=facts.trace,
            prompt=facts.prompt,
            provider=facts.provider,
            model=facts.model,
            parameters=facts.parameters,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=_duration_ms(facts.started_at, ended_at),
            finish_reason=facts.finish_reason,
        )
        self._runs.pop(assistant.id, None)
        return {"phase": "completed", "error_code": None}

    async def _load_valid_turn(
        self, state: TurnState, context: TurnContext
    ) -> ReservedTurn:
        _validate_state_context(state, context)
        turn = await self.dependencies.messages.find_turn(
            context.conversation_id, context.request_id
        )
        if turn is None:
            raise InvalidMessageState("turn was not reserved")
        if (
            turn.user.id != context.user_message_id
            or turn.assistant.id != context.assistant_message_id
        ):
            raise InvalidMessageState("turn message identity mismatch")
        if turn.user.status != "completed":
            raise InvalidMessageState("reserved user message is not completed")
        return turn

    def _should_flush(self, facts: _RunFacts, now: datetime) -> bool:
        elapsed = 0.0
        if facts.last_flush_at is not None:
            elapsed = (now - facts.last_flush_at).total_seconds()
        added = _generated_characters(facts) - facts.flushed_characters
        return (
            elapsed >= self.dependencies.flush_interval_seconds
            or added >= self.dependencies.flush_character_threshold
        )


def _validate_state_context(state: TurnState, context: TurnContext) -> None:
    expected = {
        "user_id": context.user_id,
        "conversation_id": context.conversation_id,
        "request_id": context.request_id,
        "user_message_id": context.user_message_id,
        "assistant_message_id": context.assistant_message_id,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise InvalidMessageState("graph state does not match runtime context")


def _terminal_update(message: Message) -> TurnState | None:
    if message.status == "completed":
        return {"phase": "completed", "error_code": None}
    if message.status == "failed":
        return {"phase": "failed", "error_code": message.error_code}
    return None


def _initial_trace(message: Message) -> dict[str, object]:
    try:
        existing = json.loads(message.trace_json)
    except (TypeError, ValueError):
        existing = {}
    return {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "nodes": list(existing.get("nodes", [])) if isinstance(existing, dict) else [],
        "model_calls": (
            list(existing.get("model_calls", [])) if isinstance(existing, dict) else []
        ),
        "stream": {
            "chunk_count": 0,
            "first_token_ms": None,
            "client_disconnected": False,
        },
        "errors": list(existing.get("errors", [])) if isinstance(existing, dict) else [],
    }


def _append_node(
    trace: dict[str, object],
    name: str,
    status: str,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    nodes = trace.setdefault("nodes", [])
    assert isinstance(nodes, list)
    nodes.append(
        {
            "name": name,
            "status": status,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_ms": _duration_ms(started_at, ended_at),
        }
    )


def _append_model_call(facts: _RunFacts) -> None:
    calls = facts.trace.setdefault("model_calls", [])
    assert isinstance(calls, list)
    usage = facts.usage or TokenUsage()
    calls.append(
        {
            "call_id": str(uuid4()),
            "provider": facts.provider,
            "model": facts.model,
            "request_parameters": facts.parameters,
            "response_metadata": facts.response_metadata,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
            "finish_reason": facts.finish_reason,
        }
    )


def _set_stream_trace(facts: _RunFacts) -> None:
    facts.trace["stream"] = {
        "chunk_count": facts.chunk_count,
        "first_token_ms": facts.first_token_ms,
        "client_disconnected": facts.publish_failed,
    }


def _trace_errors(trace: dict[str, object]) -> list[dict[str, str]]:
    errors = trace.setdefault("errors", [])
    assert isinstance(errors, list)
    return cast(list[dict[str, str]], errors)


def _serialize_message(message: BaseMessage) -> dict[str, object]:
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    role = roles.get(message.type)
    if role is None:
        raise ValueError(f"unsupported prompt message type: {message.type}")
    return cast(
        dict[str, object], redact_secrets({"role": role, "content": message.content})
    )


def _mapping_string(value: JSONValue, key: str) -> str | None:
    if isinstance(value, dict):
        return _optional_string(value.get(key))
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _generated_characters(facts: _RunFacts) -> int:
    return len(facts.content) + len(facts.reasoning)


def _merge_usage(current: TokenUsage | None, incoming: TokenUsage) -> TokenUsage:
    """Merge cumulative provider snapshots without summing or forgetting known fields."""
    current = current or TokenUsage()
    return TokenUsage(
        input_tokens=(
            incoming.input_tokens
            if incoming.input_tokens is not None
            else current.input_tokens
        ),
        output_tokens=(
            incoming.output_tokens
            if incoming.output_tokens is not None
            else current.output_tokens
        ),
        total_tokens=(
            incoming.total_tokens
            if incoming.total_tokens is not None
            else current.total_tokens
        ),
    )


def _duration_ms(started_at: datetime, ended_at: datetime) -> int:
    return max(0, int((ended_at - started_at).total_seconds() * 1000))


def _safe_error_message(error_code: str) -> str:
    if error_code == "model_error":
        return "model generation failed"
    if error_code == "process_interrupted":
        return "generation interrupted"
    return "turn failed"
