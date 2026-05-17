from chatbot.config import ChatConfig, LlmConfig
from chatbot.main import build_runtime_llms


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
