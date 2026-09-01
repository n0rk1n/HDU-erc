from chatbot.core.config import ChatConfig, LlmConfig
from chatbot.main import build_runtime_llms


def test_runtime_model_factory_accepts_chat_config_and_uses_chat_model(monkeypatch):
    captured_configs = []

    def fake_build_chat_model(config):
        captured_configs.append(config)
        return {"model": config.model}

    monkeypatch.setattr("chatbot.main.build_chat_model", fake_build_chat_model)
    chat_llm = LlmConfig(
        provider="deepseek",
        api_key="chat-key",
        model="deepseek-chat",
        temperature=0.2,
        base_url="https://api.deepseek.com/v1",
    )
    emotion_llm = LlmConfig(
        provider="openai",
        api_key="emotion-key",
        model="gpt-4o-mini",
        temperature=0.1,
    )

    chat_model, emotion_model = build_runtime_llms(
        ChatConfig(chat_llm=chat_llm, emotion_llm=emotion_llm)
    )

    assert chat_model == {"model": "deepseek-chat"}
    assert emotion_model == {"model": "gpt-4o-mini"}
    assert captured_configs == [chat_llm, emotion_llm]
