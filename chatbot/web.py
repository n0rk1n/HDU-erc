"""FastAPI Web 入口 —— 提供聊天页面、历史接口和 SSE 流式聊天接口。"""

import json
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from chatbot.chat_service import ChatEvent, ChatService
from chatbot.config import load_config
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
    init_session_history("default", records)
    chain = build_chain(chat_llm, profile_text)
    return ChatService(chain, config, emotion_llm, initial_records=records)


def _recent_messages(limit: int) -> list[dict]:
    records = load_history()
    return [
        {
            "role": record.get("role", ""),
            "content": record.get("content", ""),
            "timestamp": record.get("timestamp", ""),
        }
        for record in records[-limit:]
        if record.get("role") in {"human", "ai"}
    ]


def create_app(service_factory: Callable[[], ChatService] = build_service) -> FastAPI:
    app = FastAPI(title="Emotion Recognition Chatbot")

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
