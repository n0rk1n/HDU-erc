from uuid import UUID

import pytest

from chatbot.core.config import GraphConfig
from chatbot.emotion.feedback import append_emotion_feedback, load_emotion_feedback
from chatbot.persistence.runtime import open_persistence


@pytest.mark.asyncio
async def test_emotion_feedback_is_uuid_keyed_and_scoped_to_client_thread(tmp_path):
    config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )
    async with open_persistence(config) as handles:
        record = await append_emotion_feedback(
            handles.store,
            "client-a",
            "thread-1",
            {
                "message_id": "ai_1",
                "turn_count": 2,
                "feedback": "wrong_emotion",
                "predicted_emotion": "sad",
                "corrected_emotion": "anxious",
            },
        )
        saved = await load_emotion_feedback(handles.store, "client-a", "thread-1")
        other_client_saved = await load_emotion_feedback(handles.store, "client-b", "thread-1")
        stored_item = (await handles.store.asearch(("client-a", "emotion_feedback", "thread-1")))[0]

    assert record["feedback"] == "wrong_emotion"
    assert saved == [record]
    assert other_client_saved == []
    assert UUID(stored_item.key).version == 4


@pytest.mark.asyncio
async def test_emotion_feedback_rejects_unknown_value(tmp_path):
    config = GraphConfig(
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite3"),
        store_db_path=str(tmp_path / "store.sqlite3"),
        timeline_limit=50,
        request_history_limit=64,
        strict_msgpack=True,
        client_id_signing_secret="a" * 32,
    )
    async with open_persistence(config) as handles:
        with pytest.raises(ValueError, match="Invalid emotion feedback"):
            await append_emotion_feedback(
                handles.store,
                "client-a",
                "thread-1",
                {"feedback": "unknown"},
            )
