"""共享聊天服务 —— 协调历史写入、情感分析和 LLM 回复生成。"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from chatbot.config import ChatConfig
from chatbot.emotion import analyze_emotion
from chatbot.history import append_message
from chatbot.llm import format_emotion_context


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
    ):
        self.chain = chain
        self.config = config
        self.emotion_llm = emotion_llm
        self.session_records = list(initial_records or [])
        self.session_id = session_id
        self.turn_count = 0
        self.current_emotion = initial_emotion

    def _append_user_message(self, message: str) -> None:
        append_message("human", message)
        self.session_records.append({"role": "human", "content": message})
        self.turn_count += 1

    def _analyze_if_due(self, message: str) -> ChatEvent | None:
        if self.turn_count % self.config.emotion_interval != 0:
            return None
        emotion_result = analyze_emotion(
            self.emotion_llm,
            self.session_records[:-1],
            message,
            previous_emotion=self.current_emotion,
            turn_count=self.turn_count,
            emotion_interval=self.config.emotion_interval,
        )
        if emotion_result.success:
            self.current_emotion = emotion_result.emotion
            return ChatEvent("emotion_done", {"emotion": emotion_result.emotion})
        return ChatEvent("emotion_error", {"error": emotion_result.error})

    def _payload(self, message: str) -> dict[str, str]:
        return {
            "input": message,
            "emotion_context": format_emotion_context(self.current_emotion),
        }

    def generate_reply(self, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message must not be empty.")

        self._append_user_message(message)
        self._analyze_if_due(message)
        result = self.chain.invoke(
            self._payload(message),
            config={"configurable": {"session_id": self.session_id}},
        )
        answer = result.content if hasattr(result, "content") else str(result)
        append_message("ai", answer)
        self.session_records.append({"role": "ai", "content": answer})
        return answer

    def stream_reply(self, message: str) -> Iterator[ChatEvent]:
        message = message.strip()
        if not message:
            yield ChatEvent("error", {"message": "Message must not be empty."})
            return

        self._append_user_message(message)
        yield ChatEvent("user_message", {"role": "human", "content": message})

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
        append_message("ai", answer)
        self.session_records.append({"role": "ai", "content": answer})
        yield ChatEvent("done", {"content": answer})
