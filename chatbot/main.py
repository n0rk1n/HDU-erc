from langchain_core.runnables.history import RunnableWithMessageHistory

from chatbot.config import ConfigError, load_config
from chatbot.llm import build_chain, build_llm


def run_chat_loop(chain: RunnableWithMessageHistory) -> None:
    print("LangChain CLI chatbot (with memory)")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        try:
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default"}},
            )
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Bot: {answer}")


def main(argv=None) -> int:
    try:
        config = load_config(argv)
        llm = build_llm(config)
        chain = build_chain(llm)
        run_chat_loop(chain)
        return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nBye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
