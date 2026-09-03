from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from chatbot.core.config import ChatConfig, GraphConfig, LlmConfig
from chatbot.memory import MemoryRuntimeConfig, StoreMemoryRepository


FIXED_NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


class EventWriter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@dataclass(frozen=True)
class FakeDependencies:
    chat_model: Any
    emotion_model: Any
    chat_config: ChatConfig
    graph_config: GraphConfig
    memory_repository: StoreMemoryRepository
    memory_config: MemoryRuntimeConfig
    now: Any = lambda: FIXED_NOW


class EmotionModel:
    def __init__(self, output: str | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.inputs: list[Any] = []

    async def ainvoke(self, value, config=None, **kwargs):
        self.inputs.append(value)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.output or "")


def make_runtime(
    client_id: str = "client-a",
    request_id: str = "req-1",
    *,
    store: InMemoryStore | None = None,
):
    return SimpleNamespace(
        context={"client_id": client_id, "request_id": request_id, "locale": "zh-CN"},
        store=store or InMemoryStore(),
    )


@pytest.fixture
def writer() -> EventWriter:
    return EventWriter()


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def runtime(store):
    return make_runtime(store=store)


@pytest.fixture
def chat_config() -> ChatConfig:
    llm = LlmConfig(provider="test", api_key="test", model="test", temperature=0.0)
    return ChatConfig(chat_llm=llm, emotion_llm=llm, emotion_interval=5)


@pytest.fixture
def graph_config(tmp_path) -> GraphConfig:
    return GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=3,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )


@pytest.fixture
def emotion_model() -> EmotionModel:
    return EmotionModel(
        '{"primary_emotion":"anxious","confidence":0.91,'
        '"secondary_emotions":["afraid"],"evidence":"uncertainty",'
        '"reply_strategy":"validate first","trajectory_note":"worry increased",'
        '"safety_level":"normal"}'
    )


@pytest.fixture
def deps(chat_config, graph_config, store, emotion_model) -> FakeDependencies:
    return FakeDependencies(
        chat_model=object(),
        emotion_model=emotion_model,
        chat_config=chat_config,
        graph_config=graph_config,
        memory_repository=StoreMemoryRepository(store, now=lambda: FIXED_NOW),
        memory_config=MemoryRuntimeConfig(enabled=True, max_results=5),
    )


@pytest.fixture
def deps_with_failing_emotion(chat_config, graph_config, store) -> FakeDependencies:
    return FakeDependencies(
        chat_model=object(),
        emotion_model=EmotionModel(error=RuntimeError("provider unavailable")),
        chat_config=chat_config,
        graph_config=graph_config,
        memory_repository=StoreMemoryRepository(store, now=lambda: FIXED_NOW),
        memory_config=MemoryRuntimeConfig(enabled=True, max_results=5),
    )
