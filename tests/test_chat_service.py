from types import SimpleNamespace

from chatbot.chat_service import ChatService, ChatEvent
from chatbot.core.config import ChatConfig, LlmConfig
from chatbot.emotion import load_analysis_records
from chatbot.emotion_state import EmotionState
from chatbot.core.history import RegenerationUpdateResult
from chatbot.core.llm import get_session_history
from chatbot.memory import Memory, MemoryCandidate
from chatbot.memory_consolidation import MemoryConsolidationConfig


class FakeChain:
    def __init__(self, replies=None, error=None):
        self.payloads = []
        self.replies = list(replies or ["reply"])
        self.error = error

    def invoke(self, payload, config):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.replies.pop(0))


class FakeEmotionLlm:
    def __init__(self, output="Emotion: anxious", error=None):
        self.prompts = []
        self.output = output
        self.error = error

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.output)


class StreamingFakeChain(FakeChain):
    def stream(self, payload, config):
        self.payloads.append(payload)
        yield SimpleNamespace(content="hello")
        yield SimpleNamespace(content=" world")


class FailingStreamingFakeChain(FakeChain):
    def stream(self, payload, config):
        self.payloads.append(payload)
        yield SimpleNamespace(content="partial")
        raise RuntimeError("stream failed")


class LiveHistoryMutatingChain(FakeChain):
    def invoke(self, payload, config):
        self.payloads.append(payload)
        session_id = config["configurable"]["session_id"]
        history = get_session_history(session_id)
        answer = self.replies.pop(0)
        history.add_user_message(payload["input"])
        history.add_ai_message(answer)
        return SimpleNamespace(content=answer)


class StreamingLiveHistoryMutatingChain(FakeChain):
    def stream(self, payload, config):
        self.payloads.append(payload)
        session_id = config["configurable"]["session_id"]
        history = get_session_history(session_id)
        history.add_user_message(payload["input"])
        history.add_ai_message("partial answer")
        yield SimpleNamespace(content="partial")
        yield SimpleNamespace(content=" answer")


class FakeMemoryProvider:
    def __init__(self, memories=None, search_error=None, remember_error=None):
        self.memories = list(memories or [])
        self.search_error = search_error
        self.remember_error = remember_error
        self.searches = []
        self.remembered = []
        self.consolidation_state = {"last_turn_count": 0, "last_message_id": None}
        self.marked_consolidated = []

    def search(self, query, *, limit):
        self.searches.append((query, limit))
        if self.search_error:
            raise self.search_error
        return self.memories[:limit]

    def remember(self, candidates):
        self.remembered.append(candidates)
        if self.remember_error:
            raise self.remember_error
        return []

    def get_consolidation_state(self):
        return dict(self.consolidation_state)

    def mark_consolidated(self, *, turn_count, last_message_id):
        self.marked_consolidated.append((turn_count, last_message_id))
        self.consolidation_state = {
            "last_turn_count": turn_count,
            "last_message_id": last_message_id,
        }


def make_test_config(emotion_interval=2):
    llm_config = LlmConfig(
        provider="openai",
        api_key="test-key",
        model="test-model",
        temperature=0.7,
    )
    return ChatConfig(
        chat_llm=llm_config,
        emotion_llm=llm_config,
        emotion_interval=emotion_interval,
    )


def make_memory(content):
    return Memory(
        id="mem_1",
        content=content,
        category="preference",
        source="chat",
        confidence=0.9,
        created_at="2026-06-13T10:00:00+00:00",
        updated_at="2026-06-13T10:00:00+00:00",
        last_used_at=None,
        use_count=0,
    )


def test_generate_reply_injects_memory_context(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider([
        make_memory("用户希望回答使用中文。")
    ])

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        memory_provider=memory_provider,
        memory_max_results=5,
    )

    assert service.generate_reply("请介绍一下项目") == "reply 1"
    assert memory_provider.searches == [("请介绍一下项目", 5)]
    assert chain.payloads[0]["memory_context"] == (
        "Relevant Long-term Memory:\n"
        "- 用户希望回答使用中文。"
    )


def test_generate_reply_searches_memory_with_emotion_context(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        initial_emotion_state=EmotionState(primary_emotion="anxious"),
        memory_provider=memory_provider,
        memory_max_results=5,
    )
    service.recent_emotions = ["sad", "anxious"]

    service.generate_reply("又来了")

    assert memory_provider.searches == [
        ("又来了\nCurrent emotion: anxious\nRecent emotions: sad, anxious", 5)
    ]


def test_generate_reply_applies_crisis_safety_before_interval(monkeypatch):
    config = make_test_config(emotion_interval=5)
    chain = FakeChain(replies=["reply 1", "reply 2"])
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    assert service.generate_reply("I want to kill myself") == "reply 1"
    assert emotion_llm.prompts == []
    assert "- primary: sad" in chain.payloads[0]["emotion_context"]
    assert "- safety guidance: crisis" in chain.payloads[0]["emotion_context"]
    assert "Use immediate supportive language" in chain.payloads[0]["emotion_context"]

    assert service.generate_reply("I am preparing slides") == "reply 2"
    assert emotion_llm.prompts == []
    assert service.current_safety == {"level": "normal", "guidance": ""}
    assert "- safety guidance: crisis" not in chain.payloads[1]["emotion_context"]
    assert "Use immediate supportive language" not in chain.payloads[1]["emotion_context"]


def test_generate_reply_remembers_candidates_after_success(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["好的"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        memory_provider=memory_provider,
        memory_max_results=5,
    )

    service.generate_reply("我希望以后都用中文回答。")

    assert len(memory_provider.remembered) == 1
    assert memory_provider.remembered[0] == [
        MemoryCandidate(
            content="用户希望以后都用中文回答。",
            category="preference",
            source="chat",
            confidence=0.85,
        )
    ]


def test_generate_reply_continues_when_memory_fails(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider(search_error=RuntimeError("memory failed"))

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        memory_provider=memory_provider,
        memory_max_results=5,
    )

    assert service.generate_reply("hello") == "reply 1"
    assert chain.payloads[0]["memory_context"] == ""


def test_generate_reply_triggers_emotion_analysis_on_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=2)
    chain = FakeChain(replies=["reply 1", "reply 2"])
    emotion_llm = FakeEmotionLlm(
        output=(
            '{"primary_emotion":"anxious","confidence":0.8,'
            '"secondary_emotions":["apprehensive"],'
            '"evidence":"The user is worried.",'
            '"reply_strategy":"Be calm.",'
            '"trajectory_note":"","safety_level":"normal"}'
        )
    )
    stored_messages = []
    stored_ai_messages = []
    runtime_db = tmp_path / "runtime.sqlite3"

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: stored_ai_messages.append(content) or {
            "role": "ai",
            "content": content,
        },
    )
    monkeypatch.setattr("chatbot.emotion.RUNTIME_DB_PATH", str(runtime_db))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    assert service.generate_reply("q1") == "reply 1"
    assert service.generate_reply("q2") == "reply 2"

    assert len(emotion_llm.prompts) == 1
    assert "q1</s>reply 1</s>q2" in emotion_llm.prompts[0]
    assert chain.payloads[0]["emotion_context"] == ""
    assert "- primary: anxious" in chain.payloads[1]["emotion_context"]
    assert "- reply strategy: Be calm." in chain.payloads[1]["emotion_context"]
    assert len(load_analysis_records()) == 1
    assert stored_messages == [
        ("human", "q1"),
        ("human", "q2"),
    ]
    assert stored_ai_messages == ["reply 1", "reply 2"]


def test_generate_reply_does_not_trigger_before_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()
    runtime_db = tmp_path / "runtime.sqlite3"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )
    monkeypatch.setattr("chatbot.emotion.RUNTIME_DB_PATH", str(runtime_db))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    assert service.generate_reply("q1") == "reply 1"
    assert emotion_llm.prompts == []
    assert chain.payloads[0]["emotion_context"] == ""
    assert load_analysis_records() == []


def test_generate_reply_uses_initial_emotion_context(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        initial_emotion="sad",
    )

    assert service.generate_reply("q1") == "reply 1"
    assert "- primary: sad" in chain.payloads[0]["emotion_context"]
    assert "- confidence: 0.00" in chain.payloads[0]["emotion_context"]


def test_generate_reply_keeps_user_message_when_chat_fails(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(error=RuntimeError("chat failed"))
    emotion_llm = FakeEmotionLlm()
    stored_messages = []

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    try:
        service.generate_reply("q1")
    except RuntimeError as exc:
        assert str(exc) == "chat failed"
    else:
        raise AssertionError("Expected RuntimeError")

    assert stored_messages == [("human", "q1")]
    assert service.session_records == [{"role": "human", "content": "q1"}]


def test_generate_reply_counts_interval_from_current_runtime(monkeypatch):
    config = make_test_config(emotion_interval=2)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()
    initial_records = [{"role": "human", "content": "old q"}]

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )

    service = ChatService(chain, config, emotion_llm, initial_records=initial_records)

    assert service.generate_reply("q1") == "reply 1"
    assert emotion_llm.prompts == []
    assert service.turn_count == 1


def test_stream_reply_falls_back_to_invoke_and_writes_ai_message(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["full reply"])
    emotion_llm = FakeEmotionLlm()
    stored_messages = []

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "feedback": None,
        },
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert [(event.event, event.data) for event in events] == [
        ("user_message", {"role": "human", "content": "hello"}),
        ("token", {"content": "full reply"}),
        ("done", {"content": "full reply", "message_id": "ai_1"}),
    ]
    assert stored_messages == [("human", "hello")]


def test_stream_reply_emits_user_tokens_and_done(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()
    stored_messages = []

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "feedback": None,
        },
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = service.stream_reply("hello")

    event = next(events)
    assert (event.event, event.data) == (
        "user_message",
        {"role": "human", "content": "hello"},
    )
    assert stored_messages == [("human", "hello")]
    event = next(events)
    assert (event.event, event.data) == ("token", {"content": "hello"})
    assert stored_messages == [("human", "hello")]
    event = next(events)
    assert (event.event, event.data) == ("token", {"content": " world"})
    assert stored_messages == [("human", "hello")]
    done = next(events)

    assert (done.event, done.data) == (
        "done",
        {"content": "hello world", "message_id": "ai_1"},
    )
    assert stored_messages == [("human", "hello")]


def test_stream_reply_records_ai_session_message_with_metadata(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "timestamp": "t1",
            "feedback": None,
        },
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert events[-1].data == {"content": "hello world", "message_id": "ai_1"}
    assert service.session_records[-1] == {
        "id": "ai_1",
        "role": "ai",
        "content": "hello world",
        "timestamp": "t1",
        "feedback": None,
    }


def test_stream_reply_records_emotion_metadata_on_ai_message(monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()
    captured = {}

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)

    def fake_append_ai_message(content, **metadata):
        captured.update(metadata)
        return {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "timestamp": "t1",
            "feedback": None,
            **metadata,
        }

    monkeypatch.setattr("chatbot.chat_service.append_ai_message", fake_append_ai_message)

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert events[-1].data["turn_count"] == 1
    assert events[-1].data["emotion_state"]["primary_emotion"] == "anxious"
    assert captured["turn_count"] == 1
    assert captured["emotion_state"]["primary_emotion"] == "anxious"
    assert service.session_records[-1]["emotion_state"]["primary_emotion"] == "anxious"


def test_stream_reply_emits_emotion_status_on_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()
    runtime_db = tmp_path / "runtime.sqlite3"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "feedback": None,
        },
    )
    monkeypatch.setattr("chatbot.emotion.RUNTIME_DB_PATH", str(runtime_db))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert [event.event for event in events] == [
        "user_message",
        "emotion_start",
        "emotion_done",
        "token",
        "token",
        "done",
    ]
    assert events[2].data["emotion"] == "anxious"
    assert events[2].data["state"]["primary_emotion"] == "anxious"
    assert events[2].data["safety"] == {"level": "normal", "guidance": ""}


def test_stream_reply_applies_crisis_safety_before_interval(monkeypatch):
    config = make_test_config(emotion_interval=5)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "feedback": None,
        },
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("I want to kill myself"))

    assert [event.event for event in events] == [
        "user_message",
        "token",
        "token",
        "done",
    ]
    assert emotion_llm.prompts == []
    assert "- primary: sad" in chain.payloads[0]["emotion_context"]
    assert "- safety guidance: crisis" in chain.payloads[0]["emotion_context"]
    assert "Use immediate supportive language" in chain.payloads[0]["emotion_context"]

    events = list(service.stream_reply("I am preparing slides"))

    assert [event.event for event in events] == [
        "user_message",
        "token",
        "token",
        "done",
    ]
    assert emotion_llm.prompts == []
    assert service.current_safety == {"level": "normal", "guidance": ""}
    assert "- safety guidance: crisis" not in chain.payloads[1]["emotion_context"]
    assert "Use immediate supportive language" not in chain.payloads[1]["emotion_context"]


def test_stream_reply_applies_supportive_safety_guidance(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm(
        output=(
            '{"primary_emotion":"anxious","confidence":0.9,'
            '"secondary_emotions":["afraid"],'
            '"evidence":"The user feels stuck.",'
            '"reply_strategy":"Be calm.",'
            '"trajectory_note":"distress is rising","safety_level":"normal"}'
        )
    )
    runtime_db = tmp_path / "runtime.sqlite3"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "feedback": None,
        },
    )
    monkeypatch.setattr("chatbot.emotion.RUNTIME_DB_PATH", str(runtime_db))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("I feel completely hopeless about this."))

    assert events[2].event == "emotion_done"
    assert events[2].data["state"]["safety_level"] == "supportive"
    assert events[2].data["state"]["reply_strategy"] == (
        "Use supportive validation before practical next steps."
    )
    assert events[2].data["safety"]["level"] == "supportive"
    assert "- safety guidance: supportive" in chain.payloads[0]["emotion_context"]
    assert (
        "- reply strategy: Use supportive validation before practical next steps."
        in chain.payloads[0]["emotion_context"]
    )


def test_generate_reply_preserves_model_safety_when_local_policy_is_normal(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm(
        output=(
            '{"primary_emotion":"sad","confidence":0.7,'
            '"secondary_emotions":[],'
            '"evidence":"The model noticed quiet distress.",'
            '"reply_strategy":"Use model-provided crisis guidance.",'
            '"trajectory_note":"","safety_level":"crisis"}'
        )
    )
    runtime_db = tmp_path / "runtime.sqlite3"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"role": "ai", "content": content},
    )
    monkeypatch.setattr("chatbot.emotion.RUNTIME_DB_PATH", str(runtime_db))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    service.generate_reply("I do not know how to describe this feeling.")

    assert "- safety guidance: crisis" in chain.payloads[0]["emotion_context"]
    assert (
        "- reply strategy: Use model-provided crisis guidance."
        in chain.payloads[0]["emotion_context"]
    )


def test_stream_reply_passes_recent_emotion_candidates_to_prompt(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm(output="Emotion: disappointed")
    runtime_db = tmp_path / "runtime.sqlite3"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "feedback": None,
        },
    )
    monkeypatch.setattr("chatbot.emotion.RUNTIME_DB_PATH", str(runtime_db))

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        initial_emotion="anxious",
    )

    events = list(service.stream_reply("I thought this would go better."))

    assert events[2].data["emotion"] == "disappointed"
    assert events[2].data["state"]["primary_emotion"] == "disappointed"
    assert events[2].data["safety"] == {"level": "normal", "guidance": ""}
    assert "More likely emotion labels: anxious" in emotion_llm.prompts[0]
    assert "True emotion label: anxious" in emotion_llm.prompts[0]
    assert service.recent_emotions == ["disappointed", "anxious"]


def test_stream_reply_emits_emotion_error_and_continues(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm(output="Emotion: unknown")
    runtime_db = tmp_path / "runtime.sqlite3"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {
            "id": "ai_1",
            "role": "ai",
            "content": content,
            "feedback": None,
        },
    )
    monkeypatch.setattr("chatbot.emotion.RUNTIME_DB_PATH", str(runtime_db))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert [event.event for event in events] == [
        "user_message",
        "emotion_start",
        "emotion_error",
        "token",
        "token",
        "done",
    ]
    assert "Failed to parse" in events[2].data["error"]
    assert events[-1].data == {"content": "hello world", "message_id": "ai_1"}


def test_stream_reply_emits_error_without_ai_history_after_partial_stream(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FailingStreamingFakeChain()
    emotion_llm = FakeEmotionLlm()
    stored_messages = []

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert [(event.event, event.data) for event in events] == [
        ("user_message", {"role": "human", "content": "hello"}),
        ("token", {"content": "partial"}),
        ("error", {"message": "stream failed"}),
    ]
    assert stored_messages == [("human", "hello")]
    assert service.session_records == [{"role": "human", "content": "hello"}]


def test_stream_reply_emits_error_without_ai_history(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(error=RuntimeError("chat failed"))
    emotion_llm = FakeEmotionLlm()
    stored_messages = []

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert events[-1].event == "error"
    assert events[-1].data == {"message": "chat failed"}
    assert stored_messages == [("human", "hello")]


def test_regenerate_reply_uses_original_prompt_and_records_new_answer(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["better answer"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider([make_memory("用户喜欢简洁回答。")])
    recorded = {}

    def fake_prepare(message_id, reason):
        assert message_id == "ai_old"
        assert reason == "不准确"
        return RegenerationUpdateResult(
            "ready",
            original_message_id="ai_old",
            reason="不准确",
            original_user_message="q1",
        )

    def fake_record(message_id, reason, content):
        recorded.update({"message_id": message_id, "reason": reason, "content": content})
        return RegenerationUpdateResult(
            "updated",
            original_message_id=message_id,
            message_id="ai_new",
            content=content,
            reason=reason,
            original_user_message="q1",
        )

    monkeypatch.setattr("chatbot.chat_service.prepare_message_regeneration", fake_prepare)
    monkeypatch.setattr("chatbot.chat_service.record_message_regeneration", fake_record)

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[
            {"role": "human", "content": "q1"},
            {"id": "ai_old", "role": "ai", "content": "bad answer", "feedback": None},
        ],
        memory_provider=memory_provider,
        memory_max_results=5,
    )

    result = service.regenerate_reply("ai_old", "不准确")

    assert result.status == "updated"
    assert result.message_id == "ai_new"
    assert result.content == "better answer"
    assert chain.payloads[0]["input"] == "q1"
    assert "用户喜欢简洁回答。" in chain.payloads[0]["memory_context"]
    assert recorded == {
        "message_id": "ai_old",
        "reason": "不准确",
        "content": "better answer",
    }


def test_regenerate_reply_does_not_call_llm_when_history_rejects(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["should not be used"])
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr(
        "chatbot.chat_service.prepare_message_regeneration",
        lambda message_id, reason: RegenerationUpdateResult("already_regenerated"),
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    result = service.regenerate_reply("ai_old", "不准确")

    assert result.status == "already_regenerated"
    assert chain.payloads == []


def test_regenerate_reply_restores_live_history_and_appends_only_new_ai(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = LiveHistoryMutatingChain(replies=["better answer"])
    emotion_llm = FakeEmotionLlm()
    session_id = "regen-live-history"
    history = get_session_history(session_id)
    history.messages = []
    history.add_user_message("q1")
    history.add_ai_message("bad answer")
    history.add_user_message("q2")
    history.add_ai_message("later answer")

    monkeypatch.setattr(
        "chatbot.chat_service.prepare_message_regeneration",
        lambda message_id, reason: RegenerationUpdateResult(
            "ready",
            original_message_id=message_id,
            reason=reason,
            original_user_message="q1",
        ),
    )
    monkeypatch.setattr(
        "chatbot.chat_service.record_message_regeneration",
        lambda message_id, reason, content: RegenerationUpdateResult(
            "updated",
            original_message_id=message_id,
            message_id="ai_new",
            content=content,
            reason=reason,
            original_user_message="q1",
        ),
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        session_id=session_id,
    )

    result = service.regenerate_reply("ai_old", "不准确")

    assert result.status == "updated"
    assert [message.content for message in history.messages] == [
        "q1",
        "bad answer",
        "q2",
        "later answer",
        "better answer",
    ]
    assert [message.type for message in history.messages] == [
        "human",
        "ai",
        "human",
        "ai",
        "ai",
    ]


def test_stream_regenerated_reply_emits_token_and_done(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()

    monkeypatch.setattr(
        "chatbot.chat_service.prepare_message_regeneration",
        lambda message_id, reason: RegenerationUpdateResult(
            "ready",
            original_message_id=message_id,
            reason=reason,
            original_user_message="q1",
        ),
    )
    monkeypatch.setattr(
        "chatbot.chat_service.record_message_regeneration",
        lambda message_id, reason, content: RegenerationUpdateResult(
            "updated",
            original_message_id=message_id,
            message_id="ai_new",
            content=content,
            reason=reason,
            original_user_message="q1",
        ),
    )

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_regenerated_reply("ai_old", "不准确"))

    assert [(event.event, event.data) for event in events] == [
        ("token", {"content": "hello"}),
        ("token", {"content": " world"}),
        (
            "done",
            {
                "status": "regenerated",
                "original_message_id": "ai_old",
                "message_id": "ai_new",
                "content": "hello world",
                "reason": "不准确",
            },
        ),
    ]
    assert service.session_records[-1] == {
        "id": "ai_new",
        "role": "ai",
        "content": "hello world",
        "feedback": None,
        "regenerated_from": "ai_old",
    }


def test_stream_regenerated_reply_restores_live_history_when_closed_early(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = StreamingLiveHistoryMutatingChain()
    emotion_llm = FakeEmotionLlm()
    session_id = "stream-regen-close"
    history = get_session_history(session_id)
    history.messages = []
    history.add_user_message("q1")
    history.add_ai_message("bad answer")
    history.add_user_message("q2")
    history.add_ai_message("later answer")

    monkeypatch.setattr(
        "chatbot.chat_service.prepare_message_regeneration",
        lambda message_id, reason: RegenerationUpdateResult(
            "ready",
            original_message_id=message_id,
            reason=reason,
            original_user_message="q1",
        ),
    )
    monkeypatch.setattr(
        "chatbot.chat_service.record_message_regeneration",
        lambda message_id, reason, content: RegenerationUpdateResult(
            "updated",
            original_message_id=message_id,
            message_id="ai_new",
            content=content,
            reason=reason,
            original_user_message="q1",
        ),
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        session_id=session_id,
    )

    events = service.stream_regenerated_reply("ai_old", "不准确")
    first = next(events)
    events.close()

    assert (first.event, first.data) == ("token", {"content": "partial"})
    assert [message.content for message in history.messages] == [
        "q1",
        "bad answer",
        "q2",
        "later answer",
    ]
    assert [message.type for message in history.messages] == [
        "human",
        "ai",
        "human",
        "ai",
    ]
    assert service.session_records == []


def test_generate_reply_runs_consolidation_when_due(monkeypatch):
    config = make_test_config(emotion_interval=10)
    chain = FakeChain(replies=["reply 1", "reply 2"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider()
    ai_messages = []

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: ai_messages.append(content) or {
            "id": f"ai_{len(ai_messages) - 1}",
            "role": "ai",
            "content": content,
        },
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        memory_provider=memory_provider,
        memory_max_results=5,
        memory_consolidation_config=MemoryConsolidationConfig(
            enabled=True,
            interval=2,
            window=4,
            mode="rules",
        ),
    )

    service.generate_reply("我只是想被听见，不要急着给建议。")
    service.generate_reply("项目压力还是很大。")

    assert any(
        candidate.content == "用户希望难受时先被倾听，不要被急着建议。"
        for batch in memory_provider.remembered
        for candidate in batch
    )
    assert memory_provider.marked_consolidated == [(2, "ai_1")]


def test_generate_reply_consolidation_uses_restored_turn_count(monkeypatch):
    config = make_test_config(emotion_interval=20)
    chain = FakeChain(replies=["reply 1", "reply 2", "reply 3", "reply 4", "reply 5"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider()
    memory_provider.consolidation_state = {
        "last_turn_count": 5,
        "last_message_id": "old_ai_4",
    }
    ai_messages = []
    initial_records = []
    for index in range(5):
        initial_records.append({
            "id": f"old_human_{index}",
            "role": "human",
            "content": f"old question {index}",
        })
        initial_records.append({
            "id": f"old_ai_{index}",
            "role": "ai",
            "content": f"old answer {index}",
        })

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: ai_messages.append(content) or {
            "id": f"new_ai_{len(ai_messages) - 1}",
            "role": "ai",
            "content": content,
        },
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=initial_records,
        memory_provider=memory_provider,
        memory_max_results=5,
        memory_consolidation_config=MemoryConsolidationConfig(
            enabled=True,
            interval=5,
            window=10,
            mode="rules",
        ),
    )

    for index in range(5):
        service.generate_reply(f"new question {index}")

    assert memory_provider.marked_consolidated == [(10, "new_ai_4")]


def test_generate_reply_continues_when_consolidation_fails(monkeypatch):
    config = make_test_config(emotion_interval=10)
    chain = FakeChain(replies=["reply 1", "reply 2"])
    emotion_llm = FakeEmotionLlm()
    memory_provider = FakeMemoryProvider()

    def raise_during_consolidation(records):
        raise RuntimeError("consolidation failed")

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr(
        "chatbot.chat_service.append_ai_message",
        lambda content: {"id": "ai_id", "role": "ai", "content": content},
    )
    monkeypatch.setattr(
        "chatbot.chat_service.extract_consolidated_memory_candidates",
        raise_during_consolidation,
    )

    service = ChatService(
        chain,
        config,
        emotion_llm,
        initial_records=[],
        memory_provider=memory_provider,
        memory_max_results=5,
        memory_consolidation_config=MemoryConsolidationConfig(
            enabled=True,
            interval=1,
            window=4,
            mode="rules",
        ),
    )

    assert service.generate_reply("hello") == "reply 1"
