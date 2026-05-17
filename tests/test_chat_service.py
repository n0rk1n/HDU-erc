from types import SimpleNamespace

from chatbot.chat_service import ChatService
from chatbot.config import ChatConfig, LlmConfig


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


def test_generate_reply_triggers_emotion_analysis_on_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=2)
    chain = FakeChain(replies=["reply 1", "reply 2"])
    emotion_llm = FakeEmotionLlm()
    stored_messages = []
    analysis_file = tmp_path / "emotion_analysis.json"

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
    )
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    assert service.generate_reply("q1") == "reply 1"
    assert service.generate_reply("q2") == "reply 2"

    assert len(emotion_llm.prompts) == 1
    assert "q1</s>reply 1</s>q2" in emotion_llm.prompts[0]
    assert chain.payloads[0]["emotion_context"] == ""
    assert chain.payloads[1]["emotion_context"] == "Current detected user emotion: anxious"
    assert analysis_file.exists()
    assert stored_messages == [
        ("human", "q1"),
        ("ai", "reply 1"),
        ("human", "q2"),
        ("ai", "reply 2"),
    ]


def test_generate_reply_does_not_trigger_before_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain(replies=["reply 1"])
    emotion_llm = FakeEmotionLlm()
    analysis_file = tmp_path / "emotion_analysis.json"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    assert service.generate_reply("q1") == "reply 1"
    assert emotion_llm.prompts == []
    assert chain.payloads[0]["emotion_context"] == ""
    assert not analysis_file.exists()


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

    service = ChatService(chain, config, emotion_llm, initial_records=[])

    events = list(service.stream_reply("hello"))

    assert [(event.event, event.data) for event in events] == [
        ("user_message", {"role": "human", "content": "hello"}),
        ("token", {"content": "full reply"}),
        ("done", {"content": "full reply"}),
    ]
    assert stored_messages == [("human", "hello"), ("ai", "full reply")]


def test_stream_reply_emits_user_tokens_and_done(monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()
    stored_messages = []

    monkeypatch.setattr(
        "chatbot.chat_service.append_message",
        lambda role, content: stored_messages.append((role, content)),
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

    assert (done.event, done.data) == ("done", {"content": "hello world"})
    assert stored_messages == [("human", "hello"), ("ai", "hello world")]


def test_stream_reply_emits_emotion_status_on_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm()
    analysis_file = tmp_path / "emotion_analysis.json"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

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
    assert events[2].data == {"emotion": "anxious"}


def test_stream_reply_emits_emotion_error_and_continues(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=1)
    chain = StreamingFakeChain()
    emotion_llm = FakeEmotionLlm(output="Emotion: unknown")
    analysis_file = tmp_path / "emotion_analysis.json"

    monkeypatch.setattr("chatbot.chat_service.append_message", lambda role, content: None)
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

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
    assert events[-1].data == {"content": "hello world"}


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
