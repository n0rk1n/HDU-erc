"""Runtime helpers for the Web chatbot."""

from chatbot.core.config import ChatConfig, ConfigError, load_config
from chatbot.core.llm import build_llm


def build_runtime_llms(config: ChatConfig):
    """构建运行时 LLM 实例；若情感 LLM 复用聊天配置则共享同一实例避免重复初始化。"""
    chat_llm = build_llm(config.chat_llm)
    if config.emotion_llm == config.chat_llm:
        return chat_llm, chat_llm
    emotion_llm = build_llm(config.emotion_llm)
    return chat_llm, emotion_llm


def main(argv=None) -> int:
    """保留非交互入口，用于配置校验并提示 Web 启动方式。"""
    try:
        load_config(argv)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    print("Interactive CLI chat has moved to the Web UI.")
    print("Run: uvicorn chatbot.web:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
