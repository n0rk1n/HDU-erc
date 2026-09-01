from datetime import datetime, timedelta, timezone

from langgraph.graph import END, START, StateGraph

from chatbot.graphs.nodes.finalization import finalize_turn, replay_completed_request
from chatbot.graphs.state import ConversationState

from .conftest import EventWriter, make_runtime


def state_with_completed_requests(count: int) -> dict:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "request_id": f"req-{count}",
        "response_message_id": f"ai_req-{count}",
        "response_content": "完整回复",
        "processed_requests": {
            f"req-{index}": {
                "status": "completed",
                "response_message_id": f"ai_req-{index}",
                "content": f"reply-{index}",
                "completed_at": (start + timedelta(seconds=index)).isoformat(),
            }
            for index in range(count)
        },
        "thread_meta": {
            "thread_id": "thread-a",
            "title": "conversation",
            "created_at": start.isoformat(),
            "updated_at": start.isoformat(),
        },
        "input_message": "用户输入",
        "target_message_id": "ai_old",
        "regeneration_reason": "retry",
        "profile_answers": [{"key": "name", "answer": "A"}],
        "risk": {
            "signals": ["crisis_term"],
            "force_emotion_analysis": True,
            "explicit_crisis": False,
        },
        "profile_context": "profile",
        "memory_context": "memory",
        "safety_state": {"level": "supportive", "guidance": "guidance"},
        "response_content": "完整回复",
        "response_message_id": f"ai_req-{count}",
        "error_code": "old_error",
        "memory_warning": "memory_write_failed",
        "replay_request": True,
    }


def test_finalize_records_trims_and_clears_transient_fields(graph_config, writer):
    """Catches completed replies being truncated, unbounded, or leaked into the next turn."""
    state = state_with_completed_requests(64)

    update = finalize_turn(
        state,
        make_runtime(request_id="req-64"),
        writer,
        graph_config,
    )

    assert len(update["processed_requests"]) == 64
    assert "req-0" not in update["processed_requests"]
    assert update["processed_requests"]["req-64"]["status"] == "completed"
    assert update["processed_requests"]["req-64"]["response_message_id"] == "ai_req-64"
    assert update["processed_requests"]["req-64"]["content"] == "完整回复"
    assert update["thread_meta"]["updated_at"] == update["processed_requests"]["req-64"]["completed_at"]
    assert update["input_message"] == ""
    assert update["target_message_id"] == ""
    assert update["regeneration_reason"] == ""
    assert update["profile_answers"] == []
    assert update["risk"] == {}
    assert update["profile_context"] == ""
    assert update["memory_context"] == ""
    assert update["safety_state"] == {}
    assert update["response_content"] == ""
    assert update["response_message_id"] == ""
    assert update["error_code"] == ""
    assert update["memory_warning"] == ""
    assert update["replay_request"] is False
    assert writer.events == [
        {
            "event": "done",
            "data": {"message_id": "ai_req-64", "content": "完整回复"},
        }
    ]


def test_replay_emits_existing_full_result(writer):
    """Catches idempotent replay regenerating or returning partial content."""
    state = {
        "request_id": "req-1",
        "processed_requests": {
            "req-1": {
                "status": "completed",
                "response_message_id": "ai_req-1",
                "content": "已有回复",
                "completed_at": "2026-09-01T00:00:00+00:00",
            }
        },
    }

    update = replay_completed_request(state, make_runtime(), writer)

    assert update == {"replay_request": False}
    assert writer.events[-1] == {
        "event": "done",
        "data": {"message_id": "ai_req-1", "content": "已有回复", "replayed": True},
    }


def test_compiled_finalize_clears_retained_memory_warning(graph_config):
    """Catches finalize returning a clear value that the state schema then drops."""
    writer = EventWriter()
    runtime = make_runtime(request_id="req-1")
    builder = StateGraph(ConversationState)
    builder.add_node(
        "finalize",
        lambda state: finalize_turn(state, runtime, writer, graph_config),
    )
    builder.add_edge(START, "finalize")
    builder.add_edge("finalize", END)

    result = builder.compile().invoke(
        {
            "messages": [],
            "request_id": "req-1",
            "response_message_id": "ai_req-1",
            "response_content": "reply",
            "memory_warning": "memory_write_failed",
        }
    )

    assert result["memory_warning"] == ""
