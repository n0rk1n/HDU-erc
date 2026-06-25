from chatbot.emotion_state import EmotionState
from chatbot.memory import MemoryRuntimeConfig
from chatbot.memory_consolidation import (
    MemoryConsolidationConfig,
    build_memory_search_query,
    load_memory_consolidation_config,
)


def test_load_memory_consolidation_config_defaults(monkeypatch):
    monkeypatch.delenv("MEMORY_CONSOLIDATION_ENABLED", raising=False)
    monkeypatch.delenv("MEMORY_CONSOLIDATION_INTERVAL", raising=False)
    monkeypatch.delenv("MEMORY_CONSOLIDATION_WINDOW", raising=False)
    monkeypatch.delenv("MEMORY_CONSOLIDATION_MODE", raising=False)

    config = load_memory_consolidation_config(
        MemoryRuntimeConfig(enabled=True, db_path="data/records/memory.sqlite3", max_results=5)
    )

    assert config == MemoryConsolidationConfig(
        enabled=True,
        interval=5,
        window=12,
        mode="rules",
    )


def test_load_memory_consolidation_config_disabled_when_memory_disabled(monkeypatch):
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "true")

    config = load_memory_consolidation_config(
        MemoryRuntimeConfig(enabled=False, db_path="data/records/memory.sqlite3", max_results=5)
    )

    assert config.enabled is False


def test_load_memory_consolidation_config_accepts_custom_values(monkeypatch):
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "false")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_INTERVAL", "7")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_WINDOW", "20")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_MODE", "rules")

    config = load_memory_consolidation_config(
        MemoryRuntimeConfig(enabled=True, db_path="data/records/memory.sqlite3", max_results=5)
    )

    assert config == MemoryConsolidationConfig(
        enabled=False,
        interval=7,
        window=20,
        mode="rules",
    )


def test_load_memory_consolidation_config_falls_back_for_invalid_values(monkeypatch):
    monkeypatch.setenv("MEMORY_CONSOLIDATION_INTERVAL", "0")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_WINDOW", "not-a-number")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_MODE", "unknown")

    config = load_memory_consolidation_config(
        MemoryRuntimeConfig(enabled=True, db_path="data/records/memory.sqlite3", max_results=5)
    )

    assert config.interval == 5
    assert config.window == 12
    assert config.mode == "rules"


def test_build_memory_search_query_includes_emotion_context():
    query = build_memory_search_query(
        "又来了",
        EmotionState(primary_emotion="anxious"),
        ["sad", "anxious", "sad"],
    )

    assert "又来了" in query
    assert "Current emotion: anxious" in query
    assert "Recent emotions: sad, anxious" in query
