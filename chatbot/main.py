from langchain_core.runnables.history import RunnableWithMessageHistory

from chatbot.config import ConfigError, load_config
from chatbot.history import append_message, load_history
from chatbot.llm import build_chain, build_llm, init_session_history


def run_chat_loop(chain: RunnableWithMessageHistory) -> None:
    print("LangChain CLI chatbot (with memory)")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        append_message("human", question)

        try:
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default"}},
            )
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        append_message("ai", answer)
        print(f"Bot: {answer}")


def main(argv=None) -> int:
    try:
        config = load_config(argv)
        records = load_history()
        llm = build_llm(config)
        init_session_history("default", records)
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
