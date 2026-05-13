import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


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


def build_llm(config: ChatConfig) -> ChatOpenAI:
    kwargs = {
        "api_key": config.api_key,
        "model": config.model,
        "temperature": config.temperature,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)


def ask_once(llm: ChatOpenAI, question: str) -> str:
    response = llm.invoke(question)
    content = response.content
    if isinstance(content, str):
        return content
    return str(content)


def run_chat_loop(llm: ChatOpenAI) -> None:
    print("LangChain CLI chatbot")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        try:
            answer = ask_once(llm, question)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Bot: {answer}")


def main(argv=None) -> int:
    try:
        config = load_config(argv)
        llm = build_llm(config)
        run_chat_loop(llm)
        return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nBye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
