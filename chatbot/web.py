"""FastAPI Web 入口 —— 提供聊天页面、历史接口和 SSE 流式聊天接口。"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot.chat_service import ChatEvent, ChatService
from chatbot.config import load_config
from chatbot.emotion import load_analysis_records, successful_emotion_snapshot
from chatbot.emotion_feedback import append_emotion_feedback
from chatbot.emotion_state import EmotionState, timeline_from_records
from chatbot.history import (
    REGENERATION_REASONS,
    load_history,
    record_message_feedback,
)
from chatbot.local_memory import build_memory_provider
from chatbot.llm import build_chain, init_session_history
from chatbot.main import build_runtime_llms
from chatbot.memory import load_memory_config
from chatbot.memory_consolidation import load_memory_consolidation_config
from chatbot.profile import format_profile, load_profile, save_profile
from chatbot.profile_onboarding import (
    ONBOARDING_QUESTIONS,
    draft_profile,
    sanitize_profile,
)

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


class ChatStreamRequest(BaseModel):
    message: str


class ProfileRequest(BaseModel):
    profile: dict[str, Any]


class ProfileOnboardingAnswer(BaseModel):
    key: str
    answer: Any = ""


class ProfileDraftRequest(BaseModel):
    answers: list[ProfileOnboardingAnswer]


def _request_payload(request: BaseModel) -> dict:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


def format_sse(event: ChatEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {data}\n\n"


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


def create_app(service_factory: Callable[[], ChatService] = build_service) -> FastAPI:
    app = FastAPI(title="Emotion Recognition Chatbot")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.on_event("startup")
    def startup() -> None:
        app.state.chat_service = service_factory()
        app.state.chat_streams = {}

    def get_service() -> ChatService:
        service = getattr(app.state, "chat_service", None)
        if service is None:
            service = service_factory()
            app.state.chat_service = service
        return service

    def get_chat_streams() -> dict[str, str]:
        streams = getattr(app.state, "chat_streams", None)
        if streams is None:
            streams = {}
            app.state.chat_streams = streams
        return streams

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/history")
    def history(limit: int = Query(default=10, gt=0, le=100)):
        return {"messages": _recent_messages(limit)}

    @app.get("/api/session")
    def session(limit: int = Query(default=10, gt=0, le=100)):
        return _session_snapshot(limit)

    @app.get("/api/profile")
    def profile():
        profile_data = load_profile()
        return {"profile": profile_data, "is_empty": not bool(profile_data)}

    @app.put("/api/profile")
    def update_profile(
        request: ProfileRequest,
        service: ChatService = Depends(get_service),
    ):
        cleaned = sanitize_profile(request.profile)
        if not save_profile(cleaned):
            raise HTTPException(status_code=500, detail="Could not save profile.")
        _refresh_service_profile(service)
        return {"status": "saved", "profile": cleaned}

    @app.get("/api/profile/onboarding/questions")
    def profile_onboarding_questions():
        return {"questions": ONBOARDING_QUESTIONS}

    @app.post("/api/profile/onboarding/draft")
    def profile_onboarding_draft(
        request: ProfileDraftRequest,
        service: ChatService = Depends(get_service),
    ):
        runtime_chat_llm = _service_chat_llm(service)
        if runtime_chat_llm is None:
            raise HTTPException(status_code=500, detail="Chat LLM is unavailable.")
        answers = [_request_payload(answer) for answer in request.answers]
        return {"draft": draft_profile(runtime_chat_llm, answers)}

    @app.get("/api/emotion/timeline")
    def emotion_timeline(limit: int = Query(default=10, gt=0, le=50)):
        return {"timeline": _emotion_timeline_for_records(load_history(), limit)}

    @app.post("/api/messages/{message_id}/feedback")
    def message_feedback(message_id: str, request: FeedbackRequest):
        result = record_message_feedback(message_id, request.feedback)
        if result.status in {"updated", "already_rated"}:
            return {
                "status": result.status,
                "message_id": message_id,
                "feedback": result.feedback,
            }
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Message not found.")
        if result.status == "not_ai":
            raise HTTPException(
                status_code=400,
                detail="Feedback is only supported for AI messages.",
            )
        if result.status == "write_failed":
            raise HTTPException(status_code=500, detail="Could not save feedback.")
        raise HTTPException(status_code=400, detail="Invalid feedback.")

    @app.post("/api/emotion/feedback")
    def emotion_feedback(request: EmotionFeedbackRequest):
        try:
            record = append_emotion_feedback(_request_payload(request))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "saved", "feedback": record}

    @app.post("/api/messages/{message_id}/regenerate")
    def regenerate_message(
        message_id: str,
        request: RegenerateRequest,
        service: ChatService = Depends(get_service),
    ):
        if request.reason not in REGENERATION_REASONS:
            raise HTTPException(status_code=400, detail="Invalid regeneration reason.")

        try:
            result = service.regenerate_reply(message_id, request.reason)
        except Exception:
            raise HTTPException(status_code=500, detail="Could not regenerate message.")
        if result.status == "updated":
            return {
                "status": "regenerated",
                "original_message_id": message_id,
                "message_id": result.message_id,
                "content": result.content,
                "reason": result.reason,
            }
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Message not found.")
        if result.status == "not_ai":
            raise HTTPException(
                status_code=400,
                detail="Regeneration is only supported for AI messages.",
            )
        if result.status == "invalid_reason":
            raise HTTPException(status_code=400, detail="Invalid regeneration reason.")
        if result.status == "already_regenerated":
            raise HTTPException(status_code=409, detail="Message already regenerated.")
        if result.status == "missing_prompt":
            raise HTTPException(status_code=400, detail="Original prompt is unavailable.")
        if result.status in {"write_failed", "generation_failed"}:
            raise HTTPException(status_code=500, detail="Could not regenerate message.")
        raise HTTPException(status_code=500, detail="Could not regenerate message.")

    @app.post("/api/chat/streams")
    def create_chat_stream(request: ChatStreamRequest):
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message must not be empty.")
        stream_id = uuid4().hex
        get_chat_streams()[stream_id] = message
        return {"stream_id": stream_id}

    @app.get("/api/chat/streams/{stream_id}")
    def consume_chat_stream(stream_id: str, service: ChatService = Depends(get_service)):
        message = get_chat_streams().pop(stream_id, None)
        if message is None:
            raise HTTPException(status_code=410, detail="Chat stream is expired or already consumed.")

        def event_stream():
            for event in service.stream_reply(message):
                yield format_sse(event)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
