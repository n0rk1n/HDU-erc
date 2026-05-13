import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ChatConfig:
    api_key: str
    model: str
    temperature: float
    base_url: str | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chat with an OpenAI model through LangChain.")
    parser.add_argument("--model", help="OpenAI model name. Overrides OPENAI_MODEL.")
    parser.add_argument("--temperature", help="Sampling temperature. Overrides OPENAI_TEMPERATURE.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Overrides OPENAI_BASE_URL.")
    return parser.parse_args(argv)


def parse_temperature(raw_value: str) -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError("OPENAI_TEMPERATURE must be a number, such as 0.7.") from exc


def load_config(argv=None, *, load_env=True) -> ChatConfig:
    if load_env:
        load_dotenv()
    args = parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("OPENAI_API_KEY is required. Set it in .env or your environment.")

    model = args.model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    raw_temperature = args.temperature or os.getenv("OPENAI_TEMPERATURE") or str(DEFAULT_TEMPERATURE)
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or None
    if base_url is not None:
        base_url = base_url.strip() or None

    return ChatConfig(
        api_key=api_key,
        model=model,
        temperature=parse_temperature(raw_temperature),
        base_url=base_url,
    )
