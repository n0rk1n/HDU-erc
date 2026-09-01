import pytest
from langchain_core.messages import HumanMessage

from chatbot.graphs.nodes.input import GraphInputError, accept_turn


def test_accept_turn_adds_stable_human_message_and_increments_turn(writer, runtime):
    """Catches accepted turns retaining whitespace, unstable IDs, or a stale turn count."""
    update = accept_turn(
        {
            "operation": "turn",
            "request_id": "req-1",
            "input_message": " 你好 ",
            "turn_count": 0,
        },
        runtime,
        writer,
    )

    assert update["turn_count"] == 1
    assert update["messages"][0].id == "human_req-1"
    assert update["messages"][0].content == "你好"
    assert update["input_message"] == "你好"
    assert update["replay_request"] is False
    assert writer.events == [
        {
            "event": "user_message",
            "data": {"message_id": "human_req-1", "role": "human", "content": "你好"},
        }
    ]


def test_accept_turn_marks_completed_request_for_replay_without_duplicate_turn(writer, runtime):
    """Catches completed request retries appending the user message or incrementing twice."""
    update = accept_turn(
        {
            "request_id": "req-1",
            "input_message": "重试",
            "turn_count": 4,
            "processed_requests": {
                "req-1": {
                    "status": "completed",
                    "response_message_id": "ai_req-1",
                }
            },
        },
        runtime,
        writer,
    )

    assert update == {"replay_request": True}
    assert writer.events == []


def test_accept_turn_does_not_increment_when_stable_human_message_already_exists(writer, runtime):
    """Catches node retries counting the same durable human message as a second turn."""
    update = accept_turn(
        {
            "request_id": "req-1",
            "input_message": "你好",
            "turn_count": 1,
            "messages": [HumanMessage(id="human_req-1", content="你好")],
        },
        runtime,
        writer,
    )

    assert update == {"replay_request": False}
    assert writer.events == []


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_accept_turn_rejects_blank_message(message, writer, runtime):
    """Catches empty user turns entering durable graph state."""
    with pytest.raises(GraphInputError, match="empty_message"):
        accept_turn(
            {"request_id": "req-1", "input_message": message, "turn_count": 0},
            runtime,
            writer,
        )

    assert writer.events == []


def test_node_package_exports_task_five_interfaces():
    """Catches graph construction losing the stable public imports for Task 5 nodes."""
    from chatbot.graphs import nodes

    assert nodes.accept_turn is accept_turn
    assert callable(nodes.risk_precheck)
    assert callable(nodes.analyze_emotion_node)
    assert callable(nodes.reuse_emotion)
    assert callable(nodes.assess_safety_node)
    assert callable(nodes.load_context)
