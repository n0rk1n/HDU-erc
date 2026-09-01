"""Application-lifetime dependencies injected into graph nodes."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chatbot.core.config import ChatConfig, GraphConfig
from chatbot.memory import StoreMemoryRepository


@dataclass(frozen=True)
class NodeDependencies:
    chat_model: Any
    emotion_model: Any
    chat_config: ChatConfig
    graph_config: GraphConfig
    memory_repository: StoreMemoryRepository
    now: Callable[[], datetime]
