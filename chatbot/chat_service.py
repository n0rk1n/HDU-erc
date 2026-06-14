"""共享聊天服务 —— 协调历史写入、情感分析和 LLM 回复生成。"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from chatbot.config import ChatConfig
from chatbot.emotion import analyze_emotion
from chatbot.history import (
    RegenerationUpdateResult,
    append_ai_message,
    append_message,
    prepare_message_regeneration,
    record_message_regeneration,
)
from chatbot.llm import format_emotion_context
from chatbot.memory import DisabledMemoryProvider, MemoryProvider, format_memory_context
from chatbot.memory_extractor import extract_memory_candidates


@dataclass(frozen=True)
class ChatEvent:
    event: str
    data: dict[str, Any]


class ChatService:
    """本地单用户聊天服务，供 Web 层和测试复用。"""

    def __init__(
        self,
        chain,
        config: ChatConfig,
        emotion_llm,
        initial_records: list[dict] | None = None,
        session_id: str = "default",
        initial_emotion: str = "",
        memory_provider: MemoryProvider | None = None,
        memory_max_results: int = 5,
    ):
        self.chain = chain
        self.config = config
        self.emotion_llm = emotion_llm
        self.session_records = list(initial_records or [])
        self.session_id = session_id
        self.turn_count = 0
        self.current_emotion = initial_emotion
        self.memory_provider = memory_provider or DisabledMemoryProvider()
        self.memory_max_results = memory_max_results
        self.current_memory_context = ""
        self.recent_emotions = [initial_emotion] if initial_emotion else []

    def _append_user_message(self, message: str) -> None:
        append_message("human", message)
        self.session_records.append({"role": "human", "content": message})
        self.turn_count += 1

    def _append_ai_message(self, answer: str) -> str | None:
        record = append_ai_message(answer)
        if record is None:
            record = {"role": "ai", "content": answer}
        self.session_records.append(record)
        message_id = record.get("id")
        if isinstance(message_id, str) and message_id:
            return message_id
        return None

    def _analyze_if_due(self, message: str) -> ChatEvent | None:
        if self.turn_count % self.config.emotion_interval != 0:
            return None
        emotion_result = analyze_emotion(
            self.emotion_llm,
            self.session_records[:-1],
            message,
            previous_emotion=self.current_emotion,
            likely_emotions=self.recent_emotions,
            turn_count=self.turn_count,
            emotion_interval=self.config.emotion_interval,
        )
        if emotion_result.success:
            self.current_emotion = emotion_result.emotion
            self._remember_emotion(emotion_result.emotion)
            return ChatEvent("emotion_done", {"emotion": emotion_result.emotion})
        return ChatEvent("emotion_error", {"error": emotion_result.error})

    def _remember_emotion(self, emotion: str) -> None:
        emotion = emotion.strip()
        if not emotion:
            return
        self.recent_emotions = [
            item for item in self.recent_emotions if item != emotion
        ]
        self.recent_emotions.insert(0, emotion)
        self.recent_emotions = self.recent_emotions[:3]

    def _refresh_memory_context(self, message: str) -> None:
        try:
            memories = self.memory_provider.search(
                message,
                limit=self.memory_max_results,
            )
        except Exception as exc:
            print(f"Warning: memory search failed: {exc}")
            memories = []
        self.current_memory_context = format_memory_context(memories)

    def _remember_from_turn(self, message: str, answer: str) -> None:
        candidates = extract_memory_candidates(message, answer)
        if not candidates:
            return
        try:
            self.memory_provider.remember(candidates)
        except Exception as exc:
            print(f"Warning: memory write failed: {exc}")

    def _payload(self, message: str) -> dict[str, str]:
        return {
            "input": message,
            "memory_context": self.current_memory_context,
            "emotion_context": format_emotion_context(self.current_emotion),
        }

    def _generate_answer(self, message: str) -> str:
        result = self.chain.invoke(
            self._payload(message),
            config={"configurable": {"session_id": self.session_id}},
        )
        return result.content if hasattr(result, "content") else str(result)

    def generate_reply(self, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message must not be empty.")

        self._append_user_message(message)
        self._refresh_memory_context(message)
        self._analyze_if_due(message)
        answer = self._generate_answer(message)
        self._append_ai_message(answer)
        self._remember_from_turn(message, answer)
        return answer

    def regenerate_reply(self, message_id: str, reason: str) -> RegenerationUpdateResult:
        prepared = prepare_message_regeneration(message_id, reason)
        if prepared.status != "ready":
            return prepared

        message = prepared.original_user_message
        self._refresh_memory_context(message)
        answer = self._generate_answer(message)
        result = record_message_regeneration(message_id, reason, answer)
        if result.status == "updated":
            self._remember_from_turn(message, answer)
            self.session_records.append(
                {
                    "id": result.message_id,
                    "role": "ai",
                    "content": answer,
                    "feedback": None,
                    "regenerated_from": message_id,
                }
            )
        return result

    def stream_regenerated_reply(
        self,
        message_id: str,
        reason: str,
    ) -> Iterator[ChatEvent]:
        prepared = prepare_message_regeneration(message_id, reason)
        if prepared.status != "ready":
            yield ChatEvent("error", {"status": prepared.status})
            return

        message = prepared.original_user_message
        self._refresh_memory_context(message)
        answer_parts: list[str] = []
        try:
            stream = getattr(self.chain, "stream", None)
            if callable(stream):
                chunks = stream(
                    self._payload(message),
                    config={"configurable": {"session_id": self.session_id}},
                )
                for chunk in chunks:
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        answer_parts.append(content)
                        yield ChatEvent("token", {"content": content})
            else:
                content = self._generate_answer(message)
                answer_parts.append(content)
                yield ChatEvent("token", {"content": content})
        except Exception as exc:
            yield ChatEvent(
                "error",
                {"status": "generation_failed", "message": str(exc)},
            )
            return

        answer = "".join(answer_parts)
        result = record_message_regeneration(message_id, reason, answer)
        if result.status != "updated":
            yield ChatEvent("error", {"status": result.status})
            return

        self._remember_from_turn(message, answer)
        yield ChatEvent(
            "done",
            {
                "status": "regenerated",
                "original_message_id": message_id,
                "message_id": result.message_id,
                "content": answer,
                "reason": reason,
            },
        )

    def stream_reply(self, message: str) -> Iterator[ChatEvent]:
        message = message.strip()
        if not message:
            yield ChatEvent("error", {"message": "Message must not be empty."})
            return

        self._append_user_message(message)
        yield ChatEvent("user_message", {"role": "human", "content": message})
        self._refresh_memory_context(message)

        if self.turn_count % self.config.emotion_interval == 0:
            yield ChatEvent("emotion_start", {})
            emotion_event = self._analyze_if_due(message)
            if emotion_event is not None:
                yield emotion_event

        answer_parts: list[str] = []
        try:
            stream = getattr(self.chain, "stream", None)
            if callable(stream):
                chunks = stream(
                    self._payload(message),
                    config={"configurable": {"session_id": self.session_id}},
                )
                for chunk in chunks:
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        answer_parts.append(content)
                        yield ChatEvent("token", {"content": content})
            else:
                result = self.chain.invoke(
                    self._payload(message),
                    config={"configurable": {"session_id": self.session_id}},
                )
                content = result.content if hasattr(result, "content") else str(result)
                answer_parts.append(content)
                yield ChatEvent("token", {"content": content})
        except Exception as exc:
            yield ChatEvent("error", {"message": str(exc)})
            return

        answer = "".join(answer_parts)
        message_id = self._append_ai_message(answer)
        self._remember_from_turn(message, answer)
        data = {"content": answer}
        if message_id:
            data["message_id"] = message_id
        yield ChatEvent("done", data)
