"""FastAPI Web 入口 —— 提供聊天页面、历史接口和 SSE 流式聊天接口。"""

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, UUID4, field_validator

from chatbot.chat_service import ChatEvent, ChatService
from chatbot.core.config import load_config, load_graph_config
from chatbot.emotion import load_analysis_records, successful_emotion_snapshot
from chatbot.emotion.feedback import append_emotion_feedback
from chatbot.emotion.state import EmotionState, timeline_from_records
from chatbot.core.history import load_history
from chatbot.memory.sqlite import build_memory_provider
from chatbot.core.llm import build_chain, init_session_history
from chatbot.main import build_runtime_llms
from chatbot.graphs.runtime import ConversationRuntime, RuntimeOperationError, build_graph_runtime
from chatbot.memory import load_memory_config
from chatbot.memory.consolidation import load_memory_consolidation_config
from chatbot.models import GraphEvent
from chatbot.profile import format_profile, load_profile, save_profile
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


PUBLIC_GRAPH_EVENTS = frozenset(
    {
        "run_started",
        "user_message",
        "emotion_start",
        "emotion_done",
        "emotion_error",
        "safety",
        "token",
        "done",
        "error",
    }
)
VISIBLE_GENERATION_NODES = frozenset({"generate_reply", "generate_variant"})


def format_sse(event: ChatEvent | GraphEvent) -> str:
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
            if event is not None:
                yield event
            continue
        if chunk_type != "messages" or not isinstance(payload, (tuple, list)):
            continue
        if len(payload) != 2:
            continue
        message, metadata = payload
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("langgraph_node") not in VISIBLE_GENERATION_NODES:
            continue
        content = _message_chunk_text(getattr(message, "content", ""))
        if content:
            yield GraphEvent(event="token", data={"content": content})


def _stable_custom_event(payload: Any) -> GraphEvent | None:
    if not isinstance(payload, Mapping):
        return None
    event_name = payload.get("event")
    event_data = payload.get("data")
    if event_name not in PUBLIC_GRAPH_EVENTS or not isinstance(event_data, Mapping):
        return None
    data = dict(event_data)
    if event_name == "emotion_done":
        data.pop("safety", None)
        data.pop("safety_level", None)
        state = data.get("state")
        if isinstance(state, Mapping):
            data["state"] = {
                key: value for key, value in state.items() if key != "safety_level"
            }
    return GraphEvent(event=event_name, data=data)


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
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """Encode graph events, keep heartbeats private, and close the source on exit."""
    events = adapt_graph_stream(chunks)
    pending: asyncio.Task[GraphEvent] | None = None
    yield format_sse(
        GraphEvent(
            event="run_started",
            data={"request_id": str(request_id), "operation": operation},
        )
    )
    try:
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
            yield format_sse(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_code = getattr(exc, "code", "stream_failed")
        if not isinstance(error_code, str) or not error_code:
            error_code = "stream_failed"
        yield format_sse(GraphEvent(event="error", data={"error_code": error_code}))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await events.aclose()
        await chunks.aclose()


def build_service() -> ChatService:
    config = load_config([])
    records = load_history()
    profile_text = format_profile(load_profile())
    chat_llm, emotion_llm = build_runtime_llms(config)
    latest_emotion = _latest_emotion_for_records(records)
    latest_emotion_state = _latest_emotion_state_for_records(records)
    memory_config = load_memory_config()
    memory_consolidation_config = load_memory_consolidation_config(memory_config)
    memory_provider = build_memory_provider(memory_config)
    init_session_history("default", records)
    chain = build_chain(chat_llm, profile_text)
    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=records,
        initial_emotion=(latest_emotion or {}).get("emotion", ""),
        initial_emotion_state=latest_emotion_state,
        memory_provider=memory_provider,
        memory_max_results=memory_config.max_results,
        memory_consolidation_config=memory_consolidation_config,
    )
    service.chat_llm = chat_llm
    return service


def _service_chat_llm(service: ChatService):
    chat_llm = getattr(service, "chat_llm", None)
    if chat_llm is None or not callable(getattr(chat_llm, "invoke", None)):
        return None
    return chat_llm


def _refresh_service_profile(service: ChatService) -> None:
    runtime_chat_llm = _service_chat_llm(service)
    if runtime_chat_llm is None:
        return
    service.chain = build_chain(runtime_chat_llm, format_profile(load_profile()))


def _structured_messages(records: list[dict], limit: int) -> list[dict]:
    messages = []
    for record in records:
        if record.get("role") not in {"human", "ai"}:
            continue
        message = {
            "role": record.get("role", ""),
            "content": record.get("content", ""),
            "timestamp": record.get("timestamp", ""),
        }
        if "id" in record:
            message["id"] = record["id"]
        if "feedback" in record:
            message["feedback"] = record["feedback"]
        if "regeneration" in record:
            message["regeneration"] = record["regeneration"]
        if "regenerated_from" in record:
            message["regenerated_from"] = record["regenerated_from"]
        if "turn_count" in record:
            message["turn_count"] = record["turn_count"]
        if "emotion_state" in record:
            message["emotion_state"] = record["emotion_state"]
        if "predicted_emotion" in record:
            message["predicted_emotion"] = record["predicted_emotion"]
        messages.append(message)
    return messages[-limit:]


def _recent_messages(limit: int) -> list[dict]:
    return _structured_messages(load_history(), limit)


def _latest_emotion_for_records(records: list[dict]) -> dict | None:
    for record in reversed(load_analysis_records()):
        if not isinstance(record, dict):
            continue
        snapshot = successful_emotion_snapshot(record)
        if snapshot is None:
            continue
        if _emotion_record_matches_history(record, records, snapshot["turn_count"]):
            return snapshot
        return None
    return None


def _latest_emotion_state_for_records(records: list[dict]) -> EmotionState | None:
    for record in reversed(load_analysis_records()):
        if not isinstance(record, dict):
            continue
        snapshot = successful_emotion_snapshot(record)
        if snapshot is None:
            continue
        if not _emotion_record_matches_history(record, records, snapshot["turn_count"]):
            return None
        state_data = record.get("state")
        if isinstance(state_data, dict):
            state = EmotionState.from_mapping(state_data)
            if state is not None:
                return state
        return EmotionState(primary_emotion=snapshot["emotion"])
    return None


def _emotion_timeline_for_records(records: list[dict], limit: int) -> list[dict]:
    filtered = []
    for record in load_analysis_records():
        if not isinstance(record, dict):
            continue
        snapshot = successful_emotion_snapshot(record)
        if snapshot is None:
            continue
        if _emotion_record_matches_history(record, records, snapshot["turn_count"]):
            filtered.append(record)
    return timeline_from_records(filtered, limit)


def _emotion_record_matches_history(
    emotion_record: dict,
    records: list[dict],
    turn_count: int,
) -> bool:
    if turn_count <= 0:
        return False

    stored_context = _dialogue_context_from_record(emotion_record)
    if stored_context is None:
        return False

    expected_contents = [
        content
        for content in (part.strip() for part in stored_context.split("</s>"))
        if content
    ]
    if not expected_contents:
        return False

    history_items = [
        (record.get("role"), str(record.get("content", "")).strip())
        for record in records
        if record.get("role") in {"human", "ai"}
    ]
    history_items = [
        (role, content)
        for role, content in history_items
        if content
    ]
    return _contains_context_window(history_items, expected_contents, turn_count)


def _dialogue_context_from_record(emotion_record: dict) -> str | None:
    dialogue_context = emotion_record.get("dialogue_context")
    if isinstance(dialogue_context, str) and dialogue_context.strip():
        return dialogue_context.strip()

    input_text = emotion_record.get("input")
    if not isinstance(input_text, str) or not input_text:
        return None
    return _dialogue_context_from_prompt(input_text)


def _dialogue_context_from_prompt(input_text: str) -> str | None:
    marker = "\nDialogue context: "
    if marker in input_text:
        return input_text.rsplit(marker, 1)[1].strip()

    prefix = "Dialogue context: "
    if input_text.startswith(prefix):
        return input_text[len(prefix):].strip()

    return None


def _contains_context_window(
    history_items: list[tuple[str, str]],
    expected_contents: list[str],
    min_human_turns: int,
) -> bool:
    if len(expected_contents) > len(history_items):
        return False
    for index in range(len(history_items) - len(expected_contents) + 1):
        window = history_items[index:index + len(expected_contents)]
        contents = [content for _, content in window]
        human_turns = sum(1 for role, _ in window if role == "human")
        if contents == expected_contents and human_turns >= min_human_turns:
            return True
    return False


def _session_snapshot(limit: int) -> dict:
    records = load_history()
    return {
        "messages": _structured_messages(records, limit),
        "emotion": _latest_emotion_for_records(records),
    }


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


def create_app(service_factory: Callable[[], ChatService] = build_service) -> FastAPI:
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
        return StreamingResponse(
            _sse_graph_events(
                chunks,
                request_id=stream_request.request_id,
                operation="turn",
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
        return StreamingResponse(
            _sse_graph_events(
                chunks,
                request_id=stream_request.request_id,
                operation="regenerate",
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
