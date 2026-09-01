"""FastAPI Web 入口 —— 提供聊天页面、历史接口和 SSE 流式聊天接口。"""

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from anyio import CancelScope
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, UUID4, field_validator
from starlette.requests import ClientDisconnect

from chatbot.core.config import load_config, load_graph_config
from chatbot.emotion.feedback import append_emotion_feedback
from chatbot.graphs.runtime import ConversationRuntime, RuntimeOperationError, build_graph_runtime
from chatbot.models import GraphEvent
from chatbot.profile import load_profile, save_profile
from chatbot.profile.onboarding import (
    ONBOARDING_QUESTIONS,
    sanitize_profile,
)
from chatbot.persistence import open_persistence

STATIC_DIR = Path(__file__).parent / "static"


class FeedbackRequest(BaseModel):
    feedback: Literal["like", "dislike"]


class RegenerateRequest(BaseModel):
    reason: str


class EmotionFeedbackRequest(BaseModel):
    feedback: Literal["accurate", "too_positive", "too_negative", "wrong_emotion"]
    message_id: str = ""
    turn_count: int | None = None
    predicted_emotion: str = ""
    corrected_emotion: str = ""


class TurnStreamRequest(BaseModel):
    message: str
    request_id: UUID4

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("message must not be empty")
        return message


class RegenerationStreamRequest(BaseModel):
    request_id: UUID4
    reason: Literal["不准确", "不完整", "没有理解我的问题", "语气不合适", "其他"]


class ProfileRequest(BaseModel):
    profile: dict[str, Any]


class ProfileOnboardingAnswer(BaseModel):
    key: str
    answer: Any = ""


class ProfileDraftRequest(BaseModel):
    thread_id: str
    answers: list[ProfileOnboardingAnswer]


class ThreadCreateRequest(BaseModel):
    title: str = "新对话"


def _request_payload(request: BaseModel) -> dict:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


GRAPH_CUSTOM_EVENTS = frozenset(
    {
        "user_message",
        "emotion_start",
        "emotion_done",
        "emotion_error",
        "safety",
        "token",
        "done",
    }
)
VISIBLE_GENERATION_NODES = frozenset({"generate_reply", "generate_variant"})
REGENERATION_REASON_VALUES = frozenset(
    {"不准确", "不完整", "没有理解我的问题", "语气不合适", "其他"}
)
STREAM_ERROR_POLICIES = {
    "stream_failed": True,
    "generation_failed": True,
    "thread_not_found": False,
    "message_not_found": False,
    "missing_target": False,
    "non_ai_target": False,
    "already_regenerated": False,
    "invalid_reason": False,
    "completed_request_invalid": False,
}


def format_sse(event: GraphEvent) -> str:
    if isinstance(event, Mapping):
        event_name = event["event"]
        event_data = event["data"]
    else:
        event_name = event.event
        event_data = event.data
    data = json.dumps(event_data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


async def adapt_graph_stream(chunks: AsyncIterable[dict[str, Any]]) -> AsyncIterator[GraphEvent]:
    """Expose only stable custom events and user-visible model chunks."""
    async for chunk in chunks:
        chunk_type = chunk.get("type")
        payload = chunk.get("data")
        if chunk_type == "custom":
            event = _stable_custom_event(payload)
            if event is not None and _custom_origin_allows(
                chunk.get("ns"), event["event"]
            ):
                yield event
            continue
        if chunk_type != "messages" or not isinstance(payload, (tuple, list)):
            continue
        if len(payload) != 2:
            continue
        message, metadata = payload
        if not isinstance(metadata, Mapping):
            continue
        node_name = metadata.get("langgraph_node")
        if (
            not isinstance(node_name, str)
            or node_name not in VISIBLE_GENERATION_NODES
        ):
            continue
        content = _message_chunk_text(getattr(message, "content", ""))
        if content:
            yield GraphEvent(event="token", data={"content": content})


def _stable_custom_event(payload: Any) -> GraphEvent | None:
    if not isinstance(payload, Mapping):
        return None
    event_name = payload.get("event")
    event_data = payload.get("data")
    if (
        not isinstance(event_name, str)
        or event_name not in GRAPH_CUSTOM_EVENTS
        or not isinstance(event_data, Mapping)
    ):
        return None
    data = _validated_custom_data(event_name, event_data)
    if data is None:
        return None
    return GraphEvent(event=event_name, data=data)


def _custom_origin_allows(namespace: Any, event_name: str) -> bool:
    if namespace in (None, (), []):
        return True
    if not isinstance(namespace, (tuple, list)) or not namespace:
        return False
    child = namespace[0]
    if not isinstance(child, str):
        return False
    if child.startswith("turn:"):
        return event_name in GRAPH_CUSTOM_EVENTS
    if child.startswith("regenerate:"):
        return event_name == "done"
    return False


def _validated_custom_data(
    event_name: str,
    data: Mapping[str, Any],
) -> dict[str, Any] | None:
    if event_name == "emotion_start":
        return {}
    if event_name == "user_message":
        message_id = _nonempty_string(data.get("message_id"))
        content = _nonempty_string(data.get("content"))
        if message_id is None or content is None or data.get("role") != "human":
            return None
        return {"message_id": message_id, "role": "human", "content": content}
    if event_name == "emotion_done":
        return _validated_emotion_done(data)
    if event_name == "emotion_error":
        if data.get("error_code") != "emotion_analysis_failed":
            return None
        return {"error_code": "emotion_analysis_failed"}
    if event_name == "safety":
        level = data.get("level")
        guidance = data.get("guidance")
        if (
            not isinstance(level, str)
            or level not in {"normal", "supportive", "crisis"}
            or not isinstance(guidance, str)
        ):
            return None
        return {"level": level, "guidance": guidance}
    if event_name == "token":
        content = _nonempty_string(data.get("content"))
        return {"content": content} if content is not None else None
    if event_name == "done":
        return _validated_done(data)
    return None


def _validated_emotion_done(data: Mapping[str, Any]) -> dict[str, Any] | None:
    emotion = _nonempty_string(data.get("emotion"))
    state = data.get("state")
    if emotion is None or not isinstance(state, Mapping):
        return None
    primary = _nonempty_string(state.get("primary_emotion"))
    if primary is None:
        return None
    output_state: dict[str, Any] = {"primary_emotion": primary}
    field_types = {
        "evidence": str,
        "reply_strategy": str,
        "trajectory_note": str,
    }
    for field, expected_type in field_types.items():
        if field in state:
            value = state[field]
            if not isinstance(value, expected_type):
                return None
            output_state[field] = value
    if "confidence" in state:
        confidence = state["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None
        output_state["confidence"] = float(confidence)
    if "secondary_emotions" in state:
        secondary = state["secondary_emotions"]
        if not isinstance(secondary, list) or not all(
            isinstance(item, str) for item in secondary
        ):
            return None
        output_state["secondary_emotions"] = list(secondary)
    return {"emotion": emotion, "state": output_state}


def _validated_done(data: Mapping[str, Any]) -> dict[str, Any] | None:
    message_id = _nonempty_string(data.get("message_id"))
    content = _nonempty_string(data.get("content"))
    if message_id is None or content is None:
        return None
    output: dict[str, Any] = {"message_id": message_id, "content": content}
    if "replayed" in data:
        if not isinstance(data["replayed"], bool):
            return None
        output["replayed"] = data["replayed"]
    if "reason" in data:
        reason = data["reason"]
        if not isinstance(reason, str) or reason not in REGENERATION_REASON_VALUES:
            return None
        output["reason"] = reason
    if "regenerated" in data:
        if not isinstance(data["regenerated"], bool):
            return None
        output["regenerated"] = data["regenerated"]
    return output


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _message_chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


async def _sse_graph_events(
    chunks: AsyncIterator[dict[str, Any]],
    *,
    request_id: UUID,
    operation: Literal["turn", "regenerate"],
    thread_id: str,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """Encode graph events, keep heartbeats private, and close the source on exit."""
    events = adapt_graph_stream(chunks)
    pending: asyncio.Task[GraphEvent] | None = None
    done_event: GraphEvent | None = None
    try:
        yield format_sse(
            GraphEvent(
                event="run_started",
                data={
                    "request_id": str(request_id),
                    "operation": operation,
                    "thread_id": thread_id,
                },
            )
        )
        while True:
            pending = asyncio.create_task(anext(events))
            while True:
                done, _ = await asyncio.wait({pending}, timeout=heartbeat_seconds)
                if done:
                    break
                yield ": heartbeat\n\n"
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            finally:
                pending = None
            if event["event"] == "done":
                done_event = event
            elif done_event is None:
                yield format_sse(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        yield format_sse(_public_stream_error(exc))
    else:
        if done_event is not None:
            yield format_sse(done_event)
    finally:
        with CancelScope(shield=True):
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await events.aclose()
            await chunks.aclose()


def _public_stream_error(exc: Exception) -> GraphEvent:
    candidate = getattr(exc, "code", "stream_failed")
    error_code = (
        candidate
        if isinstance(candidate, str) and candidate in STREAM_ERROR_POLICIES
        else "stream_failed"
    )
    return GraphEvent(
        event="error",
        data={
            "error_code": error_code,
            "retryable": STREAM_ERROR_POLICIES[error_code],
        },
    )


class ClosingStreamingResponse(StreamingResponse):
    """Always close the response body, including both ASGI disconnect paths."""

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except ClientDisconnect:
            return
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()


def _issue_client_id(secret: str) -> str:
    payload = secrets.token_bytes(32)
    signature = hmac.digest(secret.encode("utf-8"), payload, hashlib.sha256)
    token = base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")
    return f"c_{token}"


def _validate_client_id(client_id: str, secret: str) -> str:
    if not client_id.startswith("c_"):
        raise HTTPException(status_code=401, detail="invalid_client_id")
    encoded = client_id[2:]
    try:
        raw = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=401, detail="invalid_client_id") from exc
    canonical = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if not hmac.compare_digest(encoded, canonical) or len(raw) != 64:
        raise HTTPException(status_code=401, detail="invalid_client_id")
    payload, supplied = raw[:32], raw[32:]
    expected = hmac.digest(secret.encode("utf-8"), payload, hashlib.sha256)
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid_client_id")
    return client_id


def _thread_record(record) -> dict[str, str]:
    return {
        "thread_id": record.thread_id,
        "title": record.title,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _graph_message(message) -> dict[str, Any]:
    output: dict[str, Any] = {
        "role": getattr(message, "type", ""),
        "content": getattr(message, "content", ""),
    }
    if getattr(message, "id", None):
        output["id"] = message.id
    metadata = getattr(message, "additional_kwargs", {})
    for key in (
        "feedback",
        "regeneration",
        "regenerated_from",
        "turn_count",
        "emotion_state",
        "predicted_emotion",
        "safety_level",
        "original_content",
        "original_audit",
        "regeneration_reason",
        "regenerated_at",
        "regenerated",
    ):
        if key in metadata:
            output[key] = metadata[key]
    return output


def _thread_snapshot(snapshot) -> dict[str, Any]:
    values = snapshot.values
    return {
        "messages": [_graph_message(message) for message in values.get("messages", [])],
        "emotion": values.get("emotion_state"),
        "emotion_timeline": values.get("emotion_timeline", []),
        "metadata": values.get("thread_meta", {}),
    }


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        graph_config = load_graph_config()
        chat_config = load_config([])
        async with open_persistence(graph_config) as handles:
            app.state.graph_config = graph_config
            app.state.graph_store = handles.store
            app.state.graph_runtime = build_graph_runtime(
                handles, chat_config, graph_config
            )
            yield

    app = FastAPI(title="Emotion Recognition Chatbot", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(RuntimeOperationError)
    async def runtime_error_handler(request: Request, exc: RuntimeOperationError):
        if exc.code in {"thread_not_found", "message_not_found"}:
            status_code = 404
        elif exc.code in {"already_rated", "already_regenerated"}:
            status_code = 409
        elif exc.code in {"invalid_feedback", "invalid_input"}:
            status_code = 422
        else:
            status_code = 500
        return JSONResponse(status_code=status_code, content={"detail": exc.code})

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": "internal_error"})

    def graph_runtime() -> ConversationRuntime:
        runtime = getattr(app.state, "graph_runtime", None)
        if runtime is None:
            raise HTTPException(status_code=500, detail="graph_runtime_unavailable")
        return runtime

    def authenticated_client(client_id: str) -> str:
        graph_config = getattr(app.state, "graph_config", None)
        if graph_config is None:
            raise HTTPException(status_code=500, detail="graph_runtime_unavailable")
        return _validate_client_id(client_id, graph_config.client_id_signing_secret)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.post("/api/clients/bootstrap", status_code=201)
    async def bootstrap_client():
        graph_config = app.state.graph_config
        client_id = _issue_client_id(graph_config.client_id_signing_secret)
        record = await graph_runtime().acreate_thread(client_id)
        return {"client_id": client_id, "thread": _thread_record(record)}

    @app.get("/api/clients/{client_id}/threads")
    async def list_threads(client_id: str):
        authenticated_client(client_id)
        records = await graph_runtime().alist_threads(client_id)
        return {"threads": [_thread_record(record) for record in records]}

    @app.post("/api/clients/{client_id}/threads", status_code=201)
    async def create_thread(client_id: str, request: ThreadCreateRequest):
        authenticated_client(client_id)
        record = await graph_runtime().acreate_thread(
            client_id, title=request.title.strip() or "新对话"
        )
        return {"thread": _thread_record(record)}

    @app.get("/api/clients/{client_id}/threads/{thread_id}")
    async def get_thread(client_id: str, thread_id: str):
        authenticated_client(client_id)
        snapshot = await graph_runtime().aget_state(client_id, thread_id)
        return _thread_snapshot(snapshot)

    @app.delete("/api/clients/{client_id}/threads/{thread_id}", status_code=204)
    async def delete_thread(client_id: str, thread_id: str):
        authenticated_client(client_id)
        await graph_runtime().adelete_thread(client_id, thread_id)
        return Response(status_code=204)

    @app.get("/api/clients/{client_id}/profile")
    async def get_profile(client_id: str):
        authenticated_client(client_id)
        profile = await load_profile(app.state.graph_store, client_id)
        return {"profile": profile, "is_empty": not bool(profile)}

    @app.put("/api/clients/{client_id}/profile")
    async def update_profile(client_id: str, request: ProfileRequest):
        authenticated_client(client_id)
        cleaned = sanitize_profile(request.profile)
        await save_profile(app.state.graph_store, client_id, cleaned)
        return {"status": "saved", "profile": cleaned}

    @app.post("/api/clients/{client_id}/profile/draft")
    async def profile_draft(client_id: str, request: ProfileDraftRequest):
        authenticated_client(client_id)
        answers = [_request_payload(answer) for answer in request.answers]
        draft = await graph_runtime().ainvoke_profile_draft(
            client_id, request.thread_id, uuid4().hex, answers
        )
        return {"draft": draft}

    @app.patch(
        "/api/clients/{client_id}/threads/{thread_id}/messages/{message_id}/feedback"
    )
    async def message_feedback(
        client_id: str,
        thread_id: str,
        message_id: str,
        request: FeedbackRequest,
    ):
        authenticated_client(client_id)
        message = await graph_runtime().aupdate_message_feedback(
            client_id, thread_id, message_id, request.feedback
        )
        return {"message_id": message.id, "feedback": request.feedback}

    @app.post(
        "/api/clients/{client_id}/threads/{thread_id}/emotion-feedback",
        status_code=201,
    )
    async def emotion_feedback(
        client_id: str,
        thread_id: str,
        request: EmotionFeedbackRequest,
    ):
        authenticated_client(client_id)
        await graph_runtime().aget_state(client_id, thread_id)
        try:
            record = await append_emotion_feedback(
                app.state.graph_store,
                client_id,
                thread_id,
                _request_payload(request),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_emotion_feedback") from exc
        return {"status": "saved", "feedback": record}

    @app.get("/api/clients/{client_id}/threads/{thread_id}/emotion-timeline")
    async def emotion_timeline(
        client_id: str,
        thread_id: str,
        limit: int = Query(default=10, gt=0, le=50),
    ):
        authenticated_client(client_id)
        snapshot = await graph_runtime().aget_state(client_id, thread_id)
        return {"timeline": snapshot.values.get("emotion_timeline", [])[-limit:]}

    @app.post("/api/clients/{client_id}/threads/{thread_id}/messages:stream")
    async def stream_turn(
        client_id: str,
        thread_id: str,
        stream_request: TurnStreamRequest,
    ):
        authenticated_client(client_id)
        chunks = await graph_runtime().astream_turn(
            client_id,
            thread_id,
            str(stream_request.request_id),
            stream_request.message,
        )
        return ClosingStreamingResponse(
            _sse_graph_events(
                chunks,
                request_id=stream_request.request_id,
                operation="turn",
                thread_id=thread_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/api/clients/{client_id}/threads/{thread_id}/messages/{message_id}/regenerate:stream"
    )
    async def stream_regeneration(
        client_id: str,
        thread_id: str,
        message_id: str,
        stream_request: RegenerationStreamRequest,
    ):
        authenticated_client(client_id)
        chunks = await graph_runtime().astream_regeneration(
            client_id,
            thread_id,
            str(stream_request.request_id),
            message_id,
            stream_request.reason,
        )
        return ClosingStreamingResponse(
            _sse_graph_events(
                chunks,
                request_id=stream_request.request_id,
                operation="regenerate",
                thread_id=thread_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/profile/onboarding/questions")
    def profile_onboarding_questions():
        return {"questions": ONBOARDING_QUESTIONS}

    return app


app = create_app()
