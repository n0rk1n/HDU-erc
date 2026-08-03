from chatbot.emotion.state import EmotionState
from chatbot.memory import MemoryRuntimeConfig
from chatbot.memory_consolidation import (
    MemoryConsolidationConfig,
    build_memory_search_query,
    consolidation_due,
    extract_consolidated_memory_candidates,
    load_memory_consolidation_config,
    recent_consolidation_window,
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


def test_extract_consolidated_memory_candidates_support_preference():
    records = [
        {"role": "human", "content": "我只是想被听见，不要急着给建议。"},
        {"role": "ai", "content": "我在。先不急着解决。"},
    ]

    candidates = extract_consolidated_memory_candidates(records)

    assert len(candidates) == 1
    assert candidates[0].category == "preference"
    assert candidates[0].content == "用户希望难受时先被倾听，不要被急着建议。"
    assert candidates[0].confidence == 0.85


def test_extract_consolidated_memory_candidates_does_not_merge_unrelated_listening_signals():
    records = [
        {"role": "human", "content": "我只是开玩笑。"},
        {"role": "human", "content": "不要给建议。"},
    ]

    assert extract_consolidated_memory_candidates(records) == []


def test_extract_consolidated_memory_candidates_boundary():
    records = [
        {"role": "human", "content": "以后不要劝我想开点。"},
        {"role": "ai", "content": "好，我会避开这种说法。"},
    ]

    candidates = extract_consolidated_memory_candidates(records)

    assert len(candidates) == 1
    assert candidates[0].category == "boundary"
    assert candidates[0].content == "用户要求不要用劝用户想开点的方式回应。"
    assert candidates[0].confidence == 0.9


def test_extract_consolidated_memory_candidates_ignores_single_transient_feeling():
    records = [
        {"role": "human", "content": "今天有点累。"},
        {"role": "ai", "content": "辛苦了。"},
    ]

    assert extract_consolidated_memory_candidates(records) == []


def test_extract_consolidated_memory_candidates_repeated_stressor():
    records = [
        {"role": "human", "content": "最近项目压力很大。"},
        {"role": "ai", "content": "听起来一直绷着。"},
        {"role": "human", "content": "项目压力还是压得我喘不过气。"},
    ]

    candidates = extract_consolidated_memory_candidates(records)

    assert len(candidates) == 1
    assert candidates[0].category == "profile"
    assert candidates[0].content == "用户在最近对话中多次提到项目压力。"
    assert candidates[0].confidence == 0.75


def test_consolidation_due_uses_interval():
    config = MemoryConsolidationConfig(enabled=True, interval=5, window=12, mode="rules")

    assert consolidation_due(config, turn_count=5, last_turn_count=0) is True
    assert consolidation_due(config, turn_count=9, last_turn_count=5) is False
    assert consolidation_due(config, turn_count=10, last_turn_count=5) is True


def test_consolidation_due_is_false_when_disabled():
    config = MemoryConsolidationConfig(enabled=False, interval=5, window=12, mode="rules")

    assert consolidation_due(config, turn_count=10, last_turn_count=0) is False


def test_recent_consolidation_window_filters_human_and_ai_after_checkpoint():
    records = [
        {"id": "m1", "role": "system", "content": "ignore"},
        {"id": "m2", "role": "human", "content": "old"},
        {"id": "m3", "role": "ai", "content": "old reply"},
        {"id": "m4", "role": "human", "content": "new"},
        {"id": "m5", "role": "ai", "content": "new reply"},
    ]

    window = recent_consolidation_window(records, window=3, last_message_id="m3")

    assert window == [
        {"id": "m4", "role": "human", "content": "new"},
        {"id": "m5", "role": "ai", "content": "new reply"},
    ]


def test_recent_consolidation_window_filters_empty_content():
    records = [
        {"id": "m1", "role": "human", "content": ""},
        {"id": "m2", "role": "ai", "content": "   "},
        {"id": "m3", "role": "human", "content": "real"},
    ]

    window = recent_consolidation_window(records, window=5, last_message_id=None)

    assert window == [{"id": "m3", "role": "human", "content": "real"}]


def test_recent_consolidation_window_limits_to_recent_messages():
    records = [
        {"id": "m1", "role": "human", "content": "one"},
        {"id": "m2", "role": "ai", "content": "two"},
        {"id": "m3", "role": "human", "content": "three"},
    ]

    window = recent_consolidation_window(records, window=2, last_message_id=None)

    assert [record["content"] for record in window] == ["two", "three"]
