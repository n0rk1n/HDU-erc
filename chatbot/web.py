from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chatbot.api import messages_router, users_router
from chatbot.api.messages import format_sse, stream_subscription
from chatbot.core.config import AppConfig
from chatbot.core.errors import DomainError
from chatbot.db.connection import Database, configure_connection
from chatbot.db.conversations import ConversationRepository
from chatbot.db.messages import MessageRepository
from chatbot.db.schema import initialize_schema
from chatbot.graph import NodeDependencies, compile_turn_graph
from chatbot.llm.openai_compatible import OpenAICompatibleChatModel
from chatbot.llm.types import ChatModelAdapter
from chatbot.services.identity import IdentityService
from chatbot.services.recovery import recover_interrupted_turns
from chatbot.services.turns import TurnCoordinator


_SAFE_ERRORS: dict[str, tuple[int, str]] = {
    "user_not_found": (404, "user not found"),
    "turn_in_progress": (409, "turn already in progress"),
    "model_error": (502, "model generation failed"),
    "database_error": (500, "database error"),
    "process_interrupted": (503, "service unavailable"),
}

_STATIC_DIR = Path(__file__).with_name("static")


def create_app(
    config: AppConfig | None = None,
    model: ChatModelAdapter | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime_config = config if config is not None else AppConfig.from_env()
        database = Database(runtime_config.sqlite_db_path)
        saver_connection: aiosqlite.Connection | None = None
        coordinator: TurnCoordinator | None = None
        try:
            await initialize_schema(database)
            database.path.parent.mkdir(parents=True, exist_ok=True)
            saver_connection = await aiosqlite.connect(database.path)
            await _configure_saver_connection(saver_connection)
            checkpointer = AsyncSqliteSaver(saver_connection)
            await checkpointer.setup()
            await _verify_saver_tables(saver_connection)

            messages = MessageRepository(database)
            recovery_report = await recover_interrupted_turns(messages, checkpointer)
            runtime_model = (
                model
                if model is not None
                else OpenAICompatibleChatModel(runtime_config)
            )
            graph = compile_turn_graph(
                NodeDependencies(
                    messages=messages,
                    model=runtime_model,
                    context_message_limit=runtime_config.context_message_limit,
                ),
                checkpointer,
            )
            conversations = ConversationRepository(database)
            coordinator = TurnCoordinator(
                conversations=conversations,
                messages=messages,
                graph=graph,
            )

            app.state.database = database
            app.state.identity = IdentityService(database)
            app.state.conversations = conversations
            app.state.messages = messages
            app.state.checkpointer = checkpointer
            app.state.coordinator = coordinator
            app.state.recovery_report = recovery_report
            yield
        finally:
            shutdown_error: BaseException | None = None
            if coordinator is not None:
                try:
                    await coordinator.shutdown(runtime_config.llm_timeout_seconds)
                except BaseException as error:
                    shutdown_error = error
            if saver_connection is not None:
                try:
                    await saver_connection.close()
                except BaseException as error:
                    if shutdown_error is None:
                        shutdown_error = error
            if shutdown_error is not None:
                raise shutdown_error

    app = FastAPI(title="Persistent Multi-user Chatbot", lifespan=lifespan)
    app.include_router(users_router)
    app.include_router(messages_router)

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(DomainError, _domain_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)
    return app


async def _configure_saver_connection(connection: aiosqlite.Connection) -> None:
    await configure_connection(connection)
    await connection.commit()


async def _verify_saver_tables(connection: aiosqlite.Connection) -> None:
    cursor = await connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name IN ('checkpoints', 'writes')
        """
    )
    tables = {row[0] for row in await cursor.fetchall()}
    if tables != {"checkpoints", "writes"}:
        raise RuntimeError("checkpoint schema setup failed")


async def _validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    return _error_response(
        400,
        "invalid_identifier_or_message",
        "invalid identifier or message",
    )


async def _domain_error_handler(request: Request, error: DomainError) -> JSONResponse:
    if error.code in {"invalid_identifier", "invalid_message"}:
        return _error_response(
            400,
            "invalid_identifier_or_message",
            "invalid identifier or message",
        )
    status, message = _SAFE_ERRORS.get(error.code, (500, "request failed"))
    return _error_response(status, error.code, message)


async def _unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    return _error_response(500, "internal_error", "internal server error")


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


app = create_app()


__all__ = [
    "app",
    "create_app",
    "format_sse",
    "stream_subscription",
]
