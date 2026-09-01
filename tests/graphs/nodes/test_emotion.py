import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.nodes.emotion import (
    analyze_emotion_node,
    assess_safety_node,
    reuse_emotion,
)


@pytest.mark.asyncio
async def test_analyze_emotion_updates_timeline_and_emits_done(deps, runtime, writer):
    """Catches successful analysis failing to advance its durable emotion snapshot."""
    update = await analyze_emotion_node(
        {
            "messages": [
                HumanMessage(id="human_old", content="之前一切还好"),
                AIMessage(id="ai_old", content="我在听"),
                HumanMessage(id="human_req-1", content="我很担心"),
            ],
            "turn_count": 2,
            "emotion_timeline": [
                {
                    "timestamp": "2026-08-31T10:00:00+00:00",
                    "turn_count": 1,
                    "primary_emotion": "content",
                }
            ],
            "recent_emotions": ["content"],
        },
        runtime,
        writer,
        deps=deps,
    )

    assert update["emotion_state"]["primary_emotion"] == "anxious"
    assert update["last_emotion_analysis_turn"] == 2
    assert update["recent_emotions"] == ["anxious", "content"]
    assert [item["primary_emotion"] for item in update["emotion_timeline"]] == [
        "content",
        "anxious",
    ]
    assert update["emotion_timeline"][-1]["timestamp"] == "2026-09-01T10:00:00+00:00"
    assert update["error_code"] == ""
    assert [event["event"] for event in writer.events] == ["emotion_start", "emotion_done"]
    assert "safety" not in writer.events[-1]["data"]
    assert "之前一切还好" in deps.emotion_model.inputs[0]
    assert "我在听" in deps.emotion_model.inputs[0]
    assert "我很担心" in deps.emotion_model.inputs[0]


@pytest.mark.asyncio
async def test_analyze_emotion_caps_timeline_to_graph_limit(deps, runtime, writer):
    """Catches emotion snapshots growing beyond the configured durable-state bound."""
    previous = [
        {
            "timestamp": f"2026-08-3{index}T10:00:00+00:00",
            "turn_count": index,
            "primary_emotion": "sad",
        }
        for index in range(1, 4)
    ]

    update = await analyze_emotion_node(
        {
            "messages": [HumanMessage(id="human_req-1", content="我很担心")],
            "turn_count": 4,
            "emotion_timeline": previous,
        },
        runtime,
        writer,
        deps=deps,
    )

    assert [item["turn_count"] for item in update["emotion_timeline"]] == [2, 3, 4]


@pytest.mark.asyncio
async def test_emotion_failure_reuses_previous_state_and_sets_error(
    deps_with_failing_emotion, runtime, store, writer
):
    """Catches model failures erasing prior emotion or persisting analysis side effects."""
    previous = {"primary_emotion": "sad", "confidence": 0.7}

    update = await analyze_emotion_node(
        {
            "emotion_state": previous,
            "turn_count": 2,
            "messages": [HumanMessage(id="human_req-1", content="我不知道怎么办")],
        },
        runtime,
        writer,
        deps=deps_with_failing_emotion,
    )

    assert update["emotion_state"] == previous
    assert update["error_code"] == "emotion_analysis_failed"
    assert "last_emotion_analysis_turn" not in update
    assert [event["event"] for event in writer.events] == ["emotion_start", "emotion_error"]
    assert writer.events[-1]["data"] == {"error_code": "emotion_analysis_failed"}
    assert await store.asearch(("client-a", "emotion_analysis")) == []


def test_reuse_emotion_keeps_existing_checkpoint_state_unchanged(writer, runtime):
    """Catches the reuse branch rewriting an already durable emotion snapshot."""
    assert reuse_emotion({"emotion_state": {"primary_emotion": "sad"}}, runtime, writer) == {}
    assert writer.events == []


def test_reuse_emotion_initializes_missing_state_without_an_event(writer, runtime):
    """Catches a first reuse branch leaving the downstream emotion key undefined."""
    assert reuse_emotion({}, runtime, writer) == {"emotion_state": None}
    assert writer.events == []


def test_assess_safety_keeps_explicit_crisis_and_emits_separate_event(writer, runtime):
    """Catches a calm model classification downgrading explicit first-person crisis risk."""
    risk = {
        "signals": ["crisis_term", "first_person_intent"],
        "force_emotion_analysis": True,
        "explicit_crisis": True,
    }

    update = assess_safety_node(
        {
            "input_message": "我现在想自杀",
            "emotion_state": {"primary_emotion": "content", "confidence": 0.2},
            "risk": risk,
        },
        runtime,
        writer,
    )

    assert update["safety_state"]["level"] == "crisis"
    assert writer.events == [{"event": "safety", "data": update["safety_state"]}]


def test_node_dependencies_reuses_the_injected_memory_repository(deps):
    """Catches graph construction replacing the application-wide repository and its locks."""
    production = NodeDependencies(
        chat_model=deps.chat_model,
        emotion_model=deps.emotion_model,
        chat_config=deps.chat_config,
        graph_config=deps.graph_config,
        memory_repository=deps.memory_repository,
        now=deps.now,
    )

    assert production.memory_repository is deps.memory_repository
