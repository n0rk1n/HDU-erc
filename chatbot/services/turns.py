from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from chatbot.core.errors import (
    DatabaseError,
    DomainError,
    InvalidMessageState,
    ProcessInterrupted,
    SAFE_PUBLIC_ERROR_MESSAGES,
    TurnInProgress,
    UserNotFound,
)
from chatbot.db.conversations import ConversationRepository
from chatbot.db.messages import MessageRepository
from chatbot.db.models import Message, ReservedTurn
from chatbot.graph.state import TurnContext, TurnState
from chatbot.services.events import TurnSubscription


class TurnGraph(Protocol):
    async def ainvoke(
        self,
        input: object,
        config: object = None,
        *,
        context: TurnContext,
        **kwargs: object,
    ) -> object:
        raise NotImplementedError


class TurnCoordinator:
    """Coordinate idempotent, per-user turns and detached SSE consumption."""

    def __init__(
        self,
        *,
        conversations: ConversationRepository,
        messages: MessageRepository,
        graph: TurnGraph,
        queue_capacity: int = 64,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self._conversations = conversations
        self._messages = messages
        self._graph = graph
        self._queue_capacity = queue_capacity
        self._locks: dict[int, asyncio.Lock] = {}
        self._active_requests: dict[int, str] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False

    async def open_stream(
        self, user_id: int, request_id: str, content: str
    ) -> TurnSubscription:
        _validate_request(user_id, request_id, content)
        async with self._lifecycle_lock:
            if self._closing:
                raise ProcessInterrupted("service is shutting down")

            conversation = await self._repository_call(
                self._conversations.get_default_by_user(user_id)
            )
            if conversation is None:
                raise UserNotFound()

            existing = await self._repository_call(
                self._messages.find_turn(conversation.id, request_id)
            )
            if existing is not None:
                _require_same_content(existing, content)
                replay = await self._replay(existing)
                if replay is not None:
                    return replay
                raise TurnInProgress()

            user_lock = self._locks.setdefault(user_id, asyncio.Lock())
            if user_lock.locked():
                raise TurnInProgress()
            await user_lock.acquire()

            turn: ReservedTurn | None = None
            subscription: TurnSubscription | None = None
            task: asyncio.Task[None] | None = None
            try:
                # Recheck under the user lock; the database constraint remains the
                # final guard if another service instance ever appears.
                existing = await self._repository_call(
                    self._messages.find_turn(conversation.id, request_id)
                )
                if existing is not None:
                    _require_same_content(existing, content)
                    replay = await self._replay(existing)
                    if replay is not None:
                        self._release_unused_lock(user_id, user_lock)
                        return replay
                    raise TurnInProgress()

                turn = await self._repository_call(
                    self._messages.reserve_turn(conversation.id, request_id, content)
                )
                _require_same_content(turn, content)
                if turn.assistant.status != "pending":
                    replay = await self._replay(turn)
                    if replay is not None:
                        self._release_unused_lock(user_id, user_lock)
                        return replay
                    raise TurnInProgress()

                subscription = TurnSubscription(queue_capacity=self._queue_capacity)
                context = TurnContext(
                    thread_id=conversation.thread_id,
                    user_id=user_id,
                    conversation_id=conversation.id,
                    request_id=request_id,
                    user_message_id=turn.user.id,
                    assistant_message_id=turn.assistant.id,
                    publisher=subscription,
                )
                state: TurnState = {
                    "user_id": user_id,
                    "conversation_id": conversation.id,
                    "request_id": request_id,
                    "user_message_id": turn.user.id,
                    "assistant_message_id": turn.assistant.id,
                    "phase": "reserved",
                    "error_code": None,
                }
                await subscription.publish(
                    "run_started",
                    {
                        "request_id": request_id,
                        "user_message_id": turn.user.id,
                        "assistant_message_id": turn.assistant.id,
                    },
                )
                await subscription.publish(
                    "user_message",
                    {
                        "id": turn.user.id,
                        "content": turn.user.content,
                        "sequence_no": turn.user.sequence_no,
                    },
                )
                producer = self._produce(context, state, subscription)
                try:
                    task = asyncio.create_task(
                        producer,
                        name=f"chat-turn-{user_id}-{request_id}",
                    )
                except BaseException:
                    producer.close()
                    raise

                def task_done(completed: asyncio.Task[None]) -> None:
                    self._turn_done(user_id, request_id, user_lock, completed)

                task.add_done_callback(task_done)
                self._tasks.add(task)
                self._active_requests[user_id] = request_id
                return subscription
            except BaseException as error:
                cleanup_error: BaseException | None = None
                try:
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    if turn is not None:
                        await self._fail_reserved_turn(
                            turn, "process_interrupted", "generation interrupted"
                        )
                except BaseException as failure:
                    cleanup_error = failure
                finally:
                    if task is not None:
                        self._tasks.discard(task)
                    active_request = self._active_requests.get(user_id)
                    if active_request is None or active_request == request_id:
                        if active_request == request_id:
                            self._active_requests.pop(user_id, None)
                        self._release_unused_lock(user_id, user_lock)
                    if subscription is not None:
                        subscription.detach()

                if isinstance(cleanup_error, asyncio.CancelledError):
                    raise cleanup_error
                if cleanup_error is not None and not isinstance(cleanup_error, Exception):
                    raise cleanup_error
                if isinstance(error, (asyncio.CancelledError, DomainError)):
                    raise
                if isinstance(error, Exception):
                    raise DatabaseError("turn initialization failed") from None
                raise

    async def shutdown(self, timeout_seconds: float) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds < 0:
            raise InvalidMessageState("invalid shutdown timeout")
        async with self._lifecycle_lock:
            self._closing = True
            tasks = set(self._tasks)
        if not tasks:
            return

        _, pending = await asyncio.wait(tasks, timeout=float(timeout_seconds))
        if not pending:
            return
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def _produce(
        self,
        context: TurnContext,
        state: TurnState,
        subscription: TurnSubscription,
    ) -> None:
        try:
            await self._graph.ainvoke(
                state,
                {"configurable": {"thread_id": context.thread_id}},
                context=context,
            )
            assistant = await self._messages.find_assistant(
                context.conversation_id, context.request_id
            )
            if assistant is None:
                raise InvalidMessageState("turn disappeared after graph completion")
            if assistant.status == "completed":
                await subscription.publish(
                    "done",
                    {"message": _public_message(assistant), "replayed": False},
                )
            elif assistant.status == "failed":
                await subscription.publish("error", _public_error(assistant.error_code))
            else:
                assistant = await self._fail_active(
                    context, "database_error", "turn did not reach a terminal state"
                )
                await subscription.publish("error", _public_error(assistant.error_code))
        except asyncio.CancelledError:
            assistant = await self._fail_active(
                context, "process_interrupted", "generation interrupted"
            )
            await subscription.publish("error", _public_error(assistant.error_code))
        except Exception:
            try:
                assistant = await self._fail_active(
                    context, "database_error", "turn processing failed"
                )
                error_code = assistant.error_code
            except Exception:
                error_code = "database_error"
            await subscription.publish("error", _public_error(error_code))
        finally:
            subscription.close()

    async def _fail_active(
        self, context: TurnContext, error_code: str, error_message: str
    ) -> Message:
        assistant = await self._messages.find_assistant(
            context.conversation_id, context.request_id
        )
        if assistant is None:
            raise InvalidMessageState("turn disappeared during failure handling")
        if assistant.status == "pending":
            assistant = await self._messages.mark_streaming(assistant.id)
        if assistant.status == "streaming":
            assistant = await self._messages.fail_assistant(
                assistant.id,
                error_code=error_code,
                error_message=error_message,
            )
        return assistant

    async def _fail_reserved_turn(
        self, turn: ReservedTurn, error_code: str, error_message: str
    ) -> Message:
        assistant = await self._messages.find_assistant(
            turn.assistant.conversation_id, turn.assistant.request_id
        )
        if assistant is None:
            raise InvalidMessageState("reserved turn disappeared during initialization")
        if assistant.status == "pending":
            assistant = await self._messages.mark_streaming(assistant.id)
        if assistant.status == "streaming":
            assistant = await self._messages.fail_assistant(
                assistant.id,
                error_code=error_code,
                error_message=error_message,
            )
        return assistant

    async def _replay(self, turn: ReservedTurn) -> TurnSubscription | None:
        assistant = turn.assistant
        if assistant.status not in {"completed", "failed"}:
            return None
        subscription = TurnSubscription(queue_capacity=self._queue_capacity)
        if assistant.status == "completed":
            await subscription.publish(
                "run_started",
                {
                    "request_id": assistant.request_id,
                    "user_message_id": turn.user.id,
                    "assistant_message_id": assistant.id,
                },
            )
            await subscription.publish(
                "done",
                {"message": _public_message(assistant), "replayed": True},
            )
        else:
            await subscription.publish("error", _public_error(assistant.error_code))
        subscription.close()
        return subscription

    async def _repository_call(self, operation):
        try:
            return await operation
        except DomainError:
            raise
        except Exception:
            raise DatabaseError() from None

    def _release_unused_lock(self, user_id: int, user_lock: asyncio.Lock) -> None:
        if self._locks.get(user_id) is not user_lock:
            return
        if user_lock.locked():
            user_lock.release()
        self._locks.pop(user_id, None)

    def _turn_done(
        self,
        user_id: int,
        request_id: str,
        user_lock: asyncio.Lock,
        task: asyncio.Task[None],
    ) -> None:
        self._tasks.discard(task)
        if self._active_requests.get(user_id) == request_id:
            self._active_requests.pop(user_id, None)
            self._release_unused_lock(user_id, user_lock)
        try:
            task.exception()
        except asyncio.CancelledError:
            pass


def _validate_request(user_id: int, request_id: str, content: str) -> None:
    if type(user_id) is not int or user_id < 1:
        raise UserNotFound()
    if not isinstance(request_id, str):
        raise InvalidMessageState("invalid request_id")
    try:
        parsed = UUID(request_id)
    except (ValueError, AttributeError):
        raise InvalidMessageState("invalid request_id") from None
    if parsed.version != 4 or str(parsed) != request_id:
        raise InvalidMessageState("invalid request_id")
    if not isinstance(content, str) or not content.strip() or "\x00" in content:
        raise InvalidMessageState("invalid message content")


def _require_same_content(turn: ReservedTurn, content: str) -> None:
    if turn.user.content != content:
        raise InvalidMessageState("request_id belongs to different content")


def _public_message(message: Message) -> dict[str, object]:
    return {
        "id": message.id,
        "request_id": message.request_id,
        "sequence_no": message.sequence_no,
        "role": message.role,
        "status": message.status,
        "content": message.content,
        "error_code": message.error_code,
        "error_message": message.error_message,
        "created_at": message.created_at,
        "updated_at": message.updated_at,
        "completed_at": message.completed_at,
    }


def _public_error(error_code: str | None) -> dict[str, object]:
    code = error_code if error_code in SAFE_PUBLIC_ERROR_MESSAGES else "database_error"
    return {"code": code, "message": SAFE_PUBLIC_ERROR_MESSAGES[code]}
