import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_EMOTION_RECOGNITION_INTERVAL = 5


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    api_key: str
    model: str
    temperature: float
    base_url: str | None = None


@dataclass(frozen=True)
class ChatConfig:
    chat_llm: LlmConfig
    emotion_llm: LlmConfig
    emotion_interval: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chat with a configurable LLM through LangChain.")
    parser.add_argument("--provider", help="Chat LLM provider. Overrides LLM_PROVIDER.")
    parser.add_argument("--model", help="Chat LLM model name. Overrides LLM_MODEL.")
    parser.add_argument("--temperature", help="Chat LLM sampling temperature. Overrides LLM_TEMPERATURE.")
    parser.add_argument("--base-url", help="OpenAI-compatible chat base URL. Overrides LLM_BASE_URL.")
    parser.add_argument("--emotion-provider", help="Emotion LLM provider. Overrides EMOTION_LLM_PROVIDER.")
    parser.add_argument("--emotion-model", help="Emotion LLM model name. Overrides EMOTION_LLM_MODEL.")
    parser.add_argument(
        "--emotion-temperature",
        help="Emotion LLM sampling temperature. Overrides EMOTION_LLM_TEMPERATURE.",
    )
    parser.add_argument(
        "--emotion-base-url",
        help="OpenAI-compatible emotion base URL. Overrides EMOTION_LLM_BASE_URL.",
    )
    parser.add_argument("--emotion-interval", help="Analyze emotion every N user turns. Overrides EMOTION_INTERVAL.")
    return parser.parse_args(argv)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _first_value(*values: str | None) -> str | None:
    for value in values:
        cleaned = _clean(value)
        if cleaned is not None:
            return cleaned
    return None


def parse_temperature(raw_value: str, name: str = "LLM_TEMPERATURE") -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, such as 0.7.") from exc


def parse_positive_int(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer.")
    return value


def _load_chat_llm_config(args) -> LlmConfig:
    api_key = _first_value(os.getenv("LLM_API_KEY"), os.getenv("OPENAI_API_KEY"))
    if api_key is None:
        raise ConfigError("LLM_API_KEY is required. Set LLM_API_KEY or legacy OPENAI_API_KEY.")

    provider = _first_value(args.provider, os.getenv("LLM_PROVIDER")) or DEFAULT_PROVIDER
    model = _first_value(args.model, os.getenv("LLM_MODEL"), os.getenv("OPENAI_MODEL")) or DEFAULT_MODEL
    raw_temperature = (
        _first_value(args.temperature, os.getenv("LLM_TEMPERATURE"), os.getenv("OPENAI_TEMPERATURE"))
        or str(DEFAULT_TEMPERATURE)
    )
    base_url = _first_value(args.base_url, os.getenv("LLM_BASE_URL"), os.getenv("OPENAI_BASE_URL"))

    return LlmConfig(
        provider=provider.lower(),
        api_key=api_key,
        model=model,
        temperature=parse_temperature(raw_temperature, "LLM_TEMPERATURE"),
        base_url=base_url,
    )


def _load_emotion_llm_config(args, chat_llm: LlmConfig) -> LlmConfig:
    emotion_model = _first_value(args.emotion_model, os.getenv("EMOTION_LLM_MODEL"))
    if emotion_model is None:
        return chat_llm

    provider = _first_value(args.emotion_provider, os.getenv("EMOTION_LLM_PROVIDER")) or chat_llm.provider
    api_key = _first_value(os.getenv("EMOTION_LLM_API_KEY")) or chat_llm.api_key
    raw_temperature = (
        _first_value(args.emotion_temperature, os.getenv("EMOTION_LLM_TEMPERATURE"))
        or str(chat_llm.temperature)
    )
    base_url = _first_value(args.emotion_base_url, os.getenv("EMOTION_LLM_BASE_URL")) or chat_llm.base_url

    return LlmConfig(
        provider=provider.lower(),
        api_key=api_key,
        model=emotion_model,
        temperature=parse_temperature(raw_temperature, "EMOTION_LLM_TEMPERATURE"),
        base_url=base_url,
    )


def load_config(argv=None, *, load_env=True) -> ChatConfig:
    if load_env:
        load_dotenv()
    args = parse_args(argv)

    chat_llm = _load_chat_llm_config(args)
    emotion_llm = _load_emotion_llm_config(args, chat_llm)
    raw_emotion_interval = (
        _first_value(args.emotion_interval, os.getenv("EMOTION_INTERVAL"))
        or str(DEFAULT_EMOTION_RECOGNITION_INTERVAL)
    )

    return ChatConfig(
        chat_llm=chat_llm,
        emotion_llm=emotion_llm,
        emotion_interval=parse_positive_int(raw_emotion_interval, "EMOTION_INTERVAL"),
    )
