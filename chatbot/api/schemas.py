from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, StrictStr, field_validator


class ResolveUserRequest(BaseModel):
    identifier: StrictStr

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if "\x00" in normalized or not 1 <= len(normalized) <= 128:
            raise ValueError("invalid identifier")
        return normalized


class SendMessageRequest(BaseModel):
    request_id: UUID
    content: StrictStr

    @field_validator("request_id", mode="before")
    @classmethod
    def require_canonical_uuid4(cls, value: object) -> UUID:
        if not isinstance(value, str):
            raise ValueError("request_id must be UUID4")
        try:
            parsed = UUID(value)
        except ValueError:
            raise ValueError("request_id must be UUID4") from None
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("request_id must be UUID4")
        return parsed

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("invalid message content")
        return value


class PublicUser(BaseModel):
    id: int
    identifier: str


class PublicConversation(BaseModel):
    id: str
    title: str


class ResolveUserResponse(BaseModel):
    user: PublicUser
    conversation: PublicConversation


class PublicEmotion(BaseModel):
    label: str
    display_label: str
    confidence: float = Field(ge=0, le=1)
    evidence: str
    source_message_id: str
    sequence_no: int
    analyzed_at: str | None


class PublicStep(BaseModel):
    id: Literal["received", "decision", "emotion", "response"]
    status: Literal["pending", "running", "completed", "failed", "skipped"]


class PublicProcessing(BaseModel):
    steps: list[PublicStep]
    emotion_status: Literal["not_started", "running", "completed", "failed", "skipped"]
    emotion_invoked: bool
    emotion: PublicEmotion | None
    elapsed_ms: int | None
    started_at: str


class PublicMessage(BaseModel):
    id: str
    request_id: str
    sequence_no: int
    role: Literal["user", "assistant"]
    status: Literal["pending", "streaming", "completed", "failed"]
    content: str
    bubbles: list[str] | None = None
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    completed_at: str | None
    processing: PublicProcessing | None = None


class MessageHistoryResponse(BaseModel):
    messages: list[PublicMessage]
    latest_emotion: PublicEmotion | None = None


class PublicError(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: PublicError


HistoryLimit = Annotated[int, Field(ge=1, le=200)]
PositiveSequence = Annotated[int, Field(ge=1)]
