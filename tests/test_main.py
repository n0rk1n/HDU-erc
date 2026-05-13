from types import SimpleNamespace

from chatbot.config import ChatConfig, LlmConfig
from chatbot.main import build_runtime_llms, run_chat_loop


class FakeChain:
    def __init__(self):
        self.payloads = []

    def invoke(self, payload, config):
        self.payloads.append(payload)
        return SimpleNamespace(content=f"reply {len(self.payloads)}")


class FakeEmotionLlm:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="Emotion: anxious")


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


def test_run_chat_loop_triggers_emotion_analysis_on_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=2)
    chain = FakeChain()
    emotion_llm = FakeEmotionLlm()
    inputs = iter(["q1", "q2", "quit"])
    stored_messages = []
    analysis_file = tmp_path / "emotion_analysis.json"

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("chatbot.main.append_message", lambda role, content: stored_messages.append((role, content)))
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    run_chat_loop(chain, config, emotion_llm, initial_records=[])

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


def test_run_chat_loop_does_not_trigger_before_interval(tmp_path, monkeypatch):
    config = make_test_config(emotion_interval=3)
    chain = FakeChain()
    emotion_llm = FakeEmotionLlm()
    inputs = iter(["q1", "quit"])
    analysis_file = tmp_path / "emotion_analysis.json"

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("chatbot.main.append_message", lambda role, content: None)
    monkeypatch.setattr("chatbot.emotion.EMOTION_ANALYSIS_FILE", str(analysis_file))

    run_chat_loop(chain, config, emotion_llm, initial_records=[])

    assert emotion_llm.prompts == []
    assert chain.payloads[0]["emotion_context"] == ""
    assert not analysis_file.exists()


def test_build_runtime_llms_reuses_chat_llm_when_emotion_config_matches(monkeypatch):
    chat_config = LlmConfig(
        provider="deepseek",
        api_key="chat-key",
        model="deepseek-chat",
        temperature=0.7,
        base_url="https://api.deepseek.com/v1",
    )
    config = ChatConfig(
        chat_llm=chat_config,
        emotion_llm=chat_config,
        emotion_interval=2,
    )
    built = []

    def fake_build_llm(llm_config):
        built.append(llm_config)
        return {"model": llm_config.model}

    monkeypatch.setattr("chatbot.main.build_llm", fake_build_llm)

    chat_llm, emotion_llm = build_runtime_llms(config)

    assert chat_llm == {"model": "deepseek-chat"}
    assert emotion_llm is chat_llm
    assert built == [chat_config]


def test_build_runtime_llms_builds_separate_emotion_llm_when_config_differs(monkeypatch):
    chat_config = LlmConfig(
        provider="deepseek",
        api_key="chat-key",
        model="deepseek-chat",
        temperature=0.7,
        base_url="https://api.deepseek.com/v1",
    )
    emotion_config = LlmConfig(
        provider="deepseek",
        api_key="chat-key",
        model="deepseek-reasoner",
        temperature=0.0,
        base_url="https://api.deepseek.com/v1",
    )
    config = ChatConfig(
        chat_llm=chat_config,
        emotion_llm=emotion_config,
        emotion_interval=2,
    )
    built = []

    def fake_build_llm(llm_config):
        built.append(llm_config)
        return {"model": llm_config.model}

    monkeypatch.setattr("chatbot.main.build_llm", fake_build_llm)

    chat_llm, emotion_llm = build_runtime_llms(config)

    assert chat_llm == {"model": "deepseek-chat"}
    assert emotion_llm == {"model": "deepseek-reasoner"}
    assert emotion_llm is not chat_llm
    assert built == [chat_config, emotion_config]
