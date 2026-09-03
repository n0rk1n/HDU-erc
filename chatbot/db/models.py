from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: int
    identifier: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Conversation:
    id: str
    user_id: int
    thread_id: str
    is_default: bool
    title: str
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    request_id: str
    sequence_no: int
    role: str
    status: str
    content: str
    reasoning_content: str | None
    trace_json: str
    prompt_json: str | None
    provider: str | None
    model: str | None
    parameters_json: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    finish_reason: str | None
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True)
class ReservedTurn:
    user: Message
    assistant: Message


@dataclass(frozen=True)
class InterruptedTurn:
    conversation_id: str
    thread_id: str
    assistant_message_id: str
