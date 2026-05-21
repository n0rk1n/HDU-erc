"""FastAPI Web 入口 —— 提供聊天页面、历史接口和 SSE 流式聊天接口。"""

import json
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from chatbot.chat_service import ChatEvent, ChatService
from chatbot.config import load_config
from chatbot.emotion import load_analysis_records, successful_emotion_snapshot
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


def _emotion_record_matches_history(
    emotion_record: dict,
    records: list[dict],
    turn_count: int,
) -> bool:
    if turn_count <= 0:
        return False

    input_text = emotion_record.get("input")
    if not isinstance(input_text, str) or not input_text:
        return False

    stored_context = _dialogue_context_from_prompt(input_text)
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
