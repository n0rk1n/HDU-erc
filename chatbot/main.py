"""Chatbot CLI entry point —— 加载配置、初始化历史与 LLM、运行交互循环。"""

from langchain_core.runnables.history import RunnableWithMessageHistory

from chatbot.config import ChatConfig, ConfigError, load_config
from chatbot.emotion import analyze_emotion
from chatbot.history import append_message, format_recent, load_history
from chatbot.llm import build_chain, build_llm, format_emotion_context, init_session_history
from chatbot.profile import format_profile, load_profile


def build_runtime_llms(config: ChatConfig):
    """构建运行时 LLM 实例；若情感 LLM 复用聊天配置则共享同一实例避免重复初始化。"""
    chat_llm = build_llm(config.chat_llm)
    if config.emotion_llm == config.chat_llm:
        return chat_llm, chat_llm
    emotion_llm = build_llm(config.emotion_llm)
    return chat_llm, emotion_llm


def run_chat_loop(
    chain: RunnableWithMessageHistory,
    config: ChatConfig,
    emotion_llm,
    *,
    initial_records: list[dict] | None = None,
) -> None:
    print("LangChain CLI chatbot (with memory)")
    print("Type a question and press Enter. Type exit or quit, or submit an empty line, to stop.")

    session_records = list(initial_records or [])
    turn_count = 0
    current_emotion = ""

    while True:
        question = input("\nYou: ").strip()
        if question.startswith("/history"):
            parts = question.split()
            n = 10
            if len(parts) > 1:
                try:
                    parsed = int(parts[1])
                    if parsed > 0:
                        n = parsed
                except ValueError:
                    pass
            all_records = load_history()
            output = format_recent(all_records, n=n)
            if output:
                print(f"\n--- 最近 {n} 条消息 ---")
                print(output)
                print("---")
            else:
                print("暂无历史消息。")
            continue
        if not question or question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        append_message("human", question)
        session_records.append({"role": "human", "content": question})
        turn_count += 1

        if turn_count % config.emotion_interval == 0:
            emotion_result = analyze_emotion(
                emotion_llm,
                session_records[:-1],
                question,
                previous_emotion=current_emotion,
                turn_count=turn_count,
                emotion_interval=config.emotion_interval,
            )
            if emotion_result.success:
                current_emotion = emotion_result.emotion

        try:
            result = chain.invoke(
                {
                    "input": question,
                    "emotion_context": format_emotion_context(current_emotion),
                },
                config={"configurable": {"session_id": "default"}},
            )
            answer = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        append_message("ai", answer)
        session_records.append({"role": "ai", "content": answer})
        print(f"Bot: {answer}")


def main(argv=None) -> int:
    """CLI 入口：加载配置 → 初始化历史/画像/LLM → 恢复会话 → 进入交互循环。"""
    try:
        config = load_config(argv)
        records = load_history()
        profile = load_profile()
        profile_text = format_profile(profile)
        chat_llm, emotion_llm = build_runtime_llms(config)
        init_session_history("default", records)
        recent = format_recent(records)
        if recent:
            print("\n--- 最近消息 ---")
            print(recent)
            print("---")
        chain = build_chain(chat_llm, profile_text)
        run_chat_loop(chain, config, emotion_llm, initial_records=records)
        return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nBye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
