from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

from langchain_core.messages import BaseMessage

from chatbot.llm.types import ModelDelta, TokenUsage


class OfflineModel:
    provider = "offline-test"
    parameters = {"model": "offline-test", "temperature": 0}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def stream(
        self, prompt: Sequence[BaseMessage]
    ) -> AsyncIterator[ModelDelta]:
        self.calls += 1
        yield ModelDelta(content="你")
        if self.fail:
            raise RuntimeError("raw model failure: sk-secret")
        yield ModelDelta(
            content="好",
            reasoning="private reasoning",
            usage=TokenUsage(input_tokens=2, output_tokens=2, total_tokens=4),
            finish_reason="stop",
            response_metadata={"authorization": "Bearer sk-secret"},
        )


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    frames: list[tuple[str, dict[str, object]]] = []
    normalized = body.replace("\r\n", "\n")
    for frame in normalized.strip().split("\n\n"):
        lines = frame.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        frames.append((event, json.loads(data)))
    return frames
