"""共享聊天服务 —— 协调历史写入、情感分析和 LLM 回复生成。"""

from collections.abc import Iterator
from dataclasses import dataclass
from threading import RLock
from typing import Any

from chatbot.config import ChatConfig
from chatbot.emotion import analyze_emotion
from chatbot.emotion_state import EmotionState
from chatbot.history import (
    RegenerationUpdateResult,
    append_ai_message,
    append_message,
    prepare_message_regeneration,
    record_message_regeneration,
)
from chatbot.llm import format_emotion_context, get_session_history
from chatbot.memory import DisabledMemoryProvider, MemoryProvider, format_memory_context
from chatbot.memory_extractor import extract_memory_candidates
from chatbot.safety import assess_safety


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
        initial_emotion_state: EmotionState | None = None,
        memory_provider: MemoryProvider | None = None,
        memory_max_results: int = 5,
    ):
        self.chain = chain
        self.config = config
        self.emotion_llm = emotion_llm
        self.session_records = list(initial_records or [])
        self.session_id = session_id
        self.turn_count = 0
        self.current_emotion = initial_emotion or (
            initial_emotion_state.primary_emotion if initial_emotion_state else ""
        )
        self.current_emotion_state = (
            initial_emotion_state
            or (EmotionState(primary_emotion=initial_emotion) if initial_emotion else None)
        )
        self.current_safety = {"level": "normal", "guidance": ""}
        self.memory_provider = memory_provider or DisabledMemoryProvider()
        self.memory_max_results = memory_max_results
        self.current_memory_context = ""
        self.recent_emotions = [self.current_emotion] if self.current_emotion else []
        self._lock = RLock()

    def _append_user_message(self, message: str) -> None:
        append_message("human", message)
        self.session_records.append({"role": "human", "content": message})
        self.turn_count += 1

    def _append_ai_message(self, answer: str) -> str | None:
        metadata = self._ai_message_metadata()
        try:
            record = append_ai_message(answer, **metadata)
        except TypeError:
            record = append_ai_message(answer)
        if record is None:
            record = {"role": "ai", "content": answer, **metadata}
        else:
            record.update(metadata)
        self.session_records.append(record)
        message_id = record.get("id")
        if isinstance(message_id, str) and message_id:
            return message_id
        return None

    def _ai_message_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.current_emotion_state is not None:
            metadata["turn_count"] = self.turn_count
            metadata["emotion_state"] = self.current_emotion_state.to_dict()
            metadata["predicted_emotion"] = self.current_emotion_state.primary_emotion
        elif self.current_emotion:
            metadata["turn_count"] = self.turn_count
            metadata["predicted_emotion"] = self.current_emotion
        return metadata

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
            self.current_emotion_state = (
                emotion_result.state or EmotionState(primary_emotion=emotion_result.emotion)
            )
            self.current_safety = assess_safety(message, self.current_emotion_state)
            if self.current_safety["level"] != "normal":
                self.current_emotion_state = EmotionState(
                    primary_emotion=self.current_emotion_state.primary_emotion,
                    confidence=self.current_emotion_state.confidence,
                    secondary_emotions=self.current_emotion_state.secondary_emotions,
                    evidence=self.current_emotion_state.evidence,
                    reply_strategy=self.current_safety["guidance"],
                    trajectory_note=self.current_emotion_state.trajectory_note,
                    safety_level=self.current_safety["level"],
                )
            self._remember_emotion(emotion_result.emotion)
            return ChatEvent(
                "emotion_done",
                {
                    "emotion": emotion_result.emotion,
                    "state": self.current_emotion_state.to_dict(),
                    "safety": self.current_safety,
                },
            )
        return ChatEvent("emotion_error", {"error": emotion_result.error})

    def _apply_turn_safety(self, message: str) -> None:
        safety = assess_safety(message, self.current_emotion_state)
        if safety["level"] == "normal":
            had_local_safety_override = self.current_safety["level"] != "normal"
            self.current_safety = {"level": "normal", "guidance": ""}
            if (
                had_local_safety_override
                and
                self.current_emotion_state is not None
                and self.current_emotion_state.safety_level != "normal"
            ):
                self.current_emotion_state = EmotionState(
                    primary_emotion=self.current_emotion_state.primary_emotion,
                    confidence=self.current_emotion_state.confidence,
                    secondary_emotions=self.current_emotion_state.secondary_emotions,
                    evidence=self.current_emotion_state.evidence,
                    reply_strategy="",
                    trajectory_note=self.current_emotion_state.trajectory_note,
                    safety_level="normal",
                )
            return
        self.current_safety = safety
        state = self.current_emotion_state or EmotionState(primary_emotion="sad")
        self.current_emotion_state = EmotionState(
            primary_emotion=state.primary_emotion,
            confidence=state.confidence,
            secondary_emotions=state.secondary_emotions,
            evidence=state.evidence,
            reply_strategy=safety["guidance"],
            trajectory_note=state.trajectory_note,
            safety_level=safety["level"],
        )

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
            "emotion_context": format_emotion_context(
                self.current_emotion_state or self.current_emotion
            ),
        }

    def _generate_answer(self, message: str) -> str:
        result = self.chain.invoke(
            self._payload(message),
            config={"configurable": {"session_id": self.session_id}},
        )
        return result.content if hasattr(result, "content") else str(result)

    def _regeneration_history_snapshot(self) -> list[Any]:
        return list(get_session_history(self.session_id).messages)

    def _restore_regeneration_history(self, snapshot: list[Any]) -> None:
        get_session_history(self.session_id).messages = list(snapshot)

    def _record_successful_regeneration(
        self,
        message_id: str,
        answer: str,
        result: RegenerationUpdateResult,
    ) -> None:
        get_session_history(self.session_id).add_ai_message(answer)
        self.session_records.append(
            {
                "id": result.message_id,
                "role": "ai",
                "content": answer,
                "feedback": None,
                "regenerated_from": message_id,
            }
        )

    def generate_reply(self, message: str) -> str:
        with self._lock:
            message = message.strip()
            if not message:
                raise ValueError("Message must not be empty.")

            self._append_user_message(message)
            self._refresh_memory_context(message)
            self._analyze_if_due(message)
            self._apply_turn_safety(message)
            answer = self._generate_answer(message)
            self._append_ai_message(answer)
            self._remember_from_turn(message, answer)
            return answer

    def regenerate_reply(self, message_id: str, reason: str) -> RegenerationUpdateResult:
        with self._lock:
            prepared = prepare_message_regeneration(message_id, reason)
            if prepared.status != "ready":
                return prepared

            message = prepared.original_user_message
            self._refresh_memory_context(message)
            history_snapshot = self._regeneration_history_snapshot()
            try:
                answer = self._generate_answer(message)
            except Exception:
                self._restore_regeneration_history(history_snapshot)
                raise
            self._restore_regeneration_history(history_snapshot)
            result = record_message_regeneration(message_id, reason, answer)
            if result.status == "updated":
                self._remember_from_turn(message, answer)
                self._record_successful_regeneration(message_id, answer, result)
            return result

    def stream_regenerated_reply(
        self,
        message_id: str,
        reason: str,
    ) -> Iterator[ChatEvent]:
        with self._lock:
            prepared = prepare_message_regeneration(message_id, reason)
            if prepared.status != "ready":
                yield ChatEvent("error", {"status": prepared.status})
                return

            message = prepared.original_user_message
            self._refresh_memory_context(message)
            history_snapshot = self._regeneration_history_snapshot()
            answer_parts: list[str] = []
            generation_finished = False
            generation_error = ""
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
                generation_finished = True
            except Exception as exc:
                generation_error = str(exc)
            finally:
                self._restore_regeneration_history(history_snapshot)

            if generation_error:
                yield ChatEvent(
                    "error",
                    {"status": "generation_failed", "message": generation_error},
                )
                return

            if not generation_finished:
                return

            answer = "".join(answer_parts)
            result = record_message_regeneration(message_id, reason, answer)
            if result.status != "updated":
                yield ChatEvent("error", {"status": result.status})
                return

            self._remember_from_turn(message, answer)
            self._record_successful_regeneration(message_id, answer, result)
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
        with self._lock:
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

            self._apply_turn_safety(message)

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
            data = {"content": answer, **self._ai_message_metadata()}
            if message_id:
                data["message_id"] = message_id
            yield ChatEvent("done", data)
