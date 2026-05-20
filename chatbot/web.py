"""FastAPI Web 入口 —— 提供聊天页面、历史接口和 SSE 流式聊天接口。"""

import json
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from chatbot.chat_service import ChatEvent, ChatService
from chatbot.config import load_config
from chatbot.emotion import load_latest_successful_emotion
from chatbot.history import load_history
from chatbot.llm import build_chain, init_session_history
from chatbot.main import build_runtime_llms
from chatbot.profile import format_profile, load_profile

STATIC_DIR = Path(__file__).parent / "static"


def format_sse(event: ChatEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {data}\n\n"


def build_service() -> ChatService:
    config = load_config([])
    records = load_history()
    profile_text = format_profile(load_profile())
    chat_llm, emotion_llm = build_runtime_llms(config)
    latest_emotion = _latest_emotion_for_records(records)
    init_session_history("default", records)
    chain = build_chain(chat_llm, profile_text)
    return ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=records,
        initial_emotion=(latest_emotion or {}).get("emotion", ""),
    )


def _structured_messages(records: list[dict], limit: int) -> list[dict]:
    messages = [
        {
            "role": record.get("role", ""),
            "content": record.get("content", ""),
            "timestamp": record.get("timestamp", ""),
        }
        for record in records
        if record.get("role") in {"human", "ai"}
    ]
    return messages[-limit:]


def _recent_messages(limit: int) -> list[dict]:
    return _structured_messages(load_history(), limit)


def _latest_emotion_for_records(records: list[dict]) -> dict | None:
    latest_emotion = load_latest_successful_emotion()
    if latest_emotion is None:
        return None

    human_turns = sum(1 for record in records if record.get("role") == "human")
    turn_count = latest_emotion.get("turn_count")
    if type(turn_count) is not int or turn_count <= 0 or turn_count > human_turns:
        return None
    return latest_emotion


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

    def get_service() -> ChatService:
        service = getattr(app.state, "chat_service", None)
        if service is None:
            service = service_factory()
            app.state.chat_service = service
        return service

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/history")
    def history(limit: int = Query(default=10, gt=0, le=100)):
        return {"messages": _recent_messages(limit)}

    @app.get("/api/session")
    def session(limit: int = Query(default=10, gt=0, le=100)):
        return _session_snapshot(limit)

    @app.get("/api/chat/stream")
    def chat_stream(message: str, service: ChatService = Depends(get_service)):
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message must not be empty.")

        def event_stream():
            for event in service.stream_reply(message):
                yield format_sse(event)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
