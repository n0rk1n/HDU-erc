from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Protocol

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import StreamingResponse

from chatbot.api.schemas import (
    HistoryLimit,
    MessageFeedbackRequest,
    MessageFeedbackResponse,
    MessageHistoryResponse,
    PositiveSequence,
    SendMessageRequest,
)
from chatbot.core.errors import SAFE_PUBLIC_ERROR_MESSAGES, UserNotFound
from chatbot.services.events import SseEvent
from chatbot.services.presentation import PresentationService


class EventSubscription(Protocol):
    def events(self) -> AsyncIterator[SseEvent]: ...

    def detach(self) -> None: ...


router = APIRouter(prefix="/api/users", tags=["messages"])


@router.get("/{user_id}/messages", response_model=MessageHistoryResponse)
async def list_messages(
    request: Request,
    user_id: Annotated[int, Path(ge=1)],
    limit: Annotated[HistoryLimit, Query()] = 100,
    before_sequence: Annotated[PositiveSequence | None, Query()] = None,
) -> MessageHistoryResponse:
    conversation = await request.app.state.conversations.get_default_by_user(user_id)
    if conversation is None:
        raise UserNotFound()
    visible = await request.app.state.messages.list_visible(
        conversation.id,
        limit=limit,
        before_sequence=before_sequence,
    )
    processing, latest = await PresentationService(request.app.state.database).snapshot(
        conversation.id, [item["request_id"] for item in visible if item["role"] == "assistant"]
    )
    return MessageHistoryResponse(
        messages=[{**_safe_history_message(item), "processing":
                   processing.get(item["request_id"]) if item["role"] == "assistant" else None}
                  for item in visible],
        latest_emotion=latest,
    )


@router.patch("/{user_id}/messages/{message_id}/feedback", response_model=MessageFeedbackResponse)
async def set_message_feedback(
    request: Request,
    payload: MessageFeedbackRequest,
    user_id: Annotated[int, Path(ge=1)],
    message_id: str,
) -> MessageFeedbackResponse:
    conversation = await request.app.state.conversations.get_default_by_user(user_id)
    if conversation is None:
        raise UserNotFound()
    await request.app.state.messages.set_feedback(conversation.id, message_id, payload.feedback)
    return MessageFeedbackResponse(message_id=message_id, feedback=payload.feedback)


@router.post("/{user_id}/messages:stream")
async def send_message_stream(
    request: Request,
    payload: SendMessageRequest,
    user_id: Annotated[int, Path(ge=1)],
) -> StreamingResponse:
    subscription = await request.app.state.coordinator.open_stream(
        user_id,
        str(payload.request_id),
        payload.content,
    )
    return StreamingResponse(
        stream_subscription(subscription),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def format_sse(event: SseEvent) -> str:
    payload = json.dumps(
        event.data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"event: {event.name}\ndata: {payload}\n\n"


async def stream_subscription(subscription: EventSubscription) -> AsyncIterator[str]:
    try:
        try:
            async for event in subscription.events():
                yield format_sse(event)
        except Exception:
            yield format_sse(
                SseEvent(
                    "error",
                    {"code": "database_error", "message": "database error"},
                )
            )
    finally:
        subscription.detach()


def _safe_history_message(message: dict[str, object | None]) -> dict[str, object | None]:
    public = dict(message)
    if public.get("status") != "failed":
        public["error_code"] = None
        public["error_message"] = None
        return public

    safe_errors = SAFE_PUBLIC_ERROR_MESSAGES
    error_code = public.get("error_code")
    if not isinstance(error_code, str) or error_code not in safe_errors:
        public["error_code"] = "message_failed"
        public["error_message"] = "message failed"
        return public
    public["error_message"] = safe_errors[error_code]
    return public
