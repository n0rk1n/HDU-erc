from chatbot.memory import (
    DisabledMemoryProvider,
    Memory,
    MemoryRuntimeConfig,
    format_memory_context,
    load_memory_config,
)


def test_format_memory_context_returns_empty_for_no_memories():
    assert format_memory_context([]) == ""


def test_format_memory_context_renders_bullets():
    memories = [
        Memory(
            id="mem_1",
            content="用户希望回答使用中文。",
            category="preference",
            source="chat",
            confidence=0.9,
            created_at="2026-06-13T10:00:00+00:00",
            updated_at="2026-06-13T10:00:00+00:00",
            last_used_at=None,
            use_count=0,
        )
    ]

    assert format_memory_context(memories) == (
        "Relevant Long-term Memory:\n"
        "- 用户希望回答使用中文。"
    )


def test_disabled_memory_provider_is_noop():
    provider = DisabledMemoryProvider()

    assert provider.search("anything", limit=5) == []
    assert provider.remember([]) == []


def test_load_memory_config_defaults(monkeypatch):
    monkeypatch.delenv("MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
    monkeypatch.delenv("MEMORY_MAX_RESULTS", raising=False)

    config = load_memory_config()

    assert config == MemoryRuntimeConfig(
        enabled=True,
        db_path="data/records/memory.sqlite3",
        max_results=5,
    )


def test_load_memory_config_accepts_false_and_custom_values(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    monkeypatch.setenv("MEMORY_DB_PATH", "tmp/memory.sqlite3")
    monkeypatch.setenv("MEMORY_MAX_RESULTS", "3")

    config = load_memory_config()

    assert config == MemoryRuntimeConfig(
        enabled=False,
        db_path="tmp/memory.sqlite3",
        max_results=3,
    )


def test_load_memory_config_clamps_invalid_max_results(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_RESULTS", "not-a-number")

    config = load_memory_config()

    assert config.max_results == 5
