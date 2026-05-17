"""LLM 链构建与会话管理 —— 封装 LangChain 的 prompt、chain 组装和 session history 存储。"""

import warnings
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory is deprecated.*")

from chatbot.config import ChatConfig, LlmConfig
from chatbot.llm_adapter import ChatModelAdapter, build_chat_model

store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_llm(config: ChatConfig | LlmConfig) -> ChatModelAdapter:
    llm_config = config.chat_llm if isinstance(config, ChatConfig) else config
    return build_chat_model(llm_config)


def ask_once(llm: ChatModelAdapter, question: str) -> str:
    response = llm.invoke(question)
    content = response.content
    if isinstance(content, str):
        return content
    return str(content)


def format_emotion_context(emotion: str) -> str:
    """将当前检测到的情绪格式化为可注入系统提示词的上下文。"""
    emotion = emotion.strip()
    if not emotion:
        return ""
    return f"Current detected user emotion: {emotion}"


def build_system_message(profile_text: str = "") -> str:
    system_message = "You are a helpful assistant."
    if profile_text:
        system_message += f"\n\nUser Profile:\n{profile_text}"
    system_message += "\n\n{emotion_context}"
    return system_message


def init_session_history(session_id: str, records: list[dict]) -> None:
    """从持久化历史恢复 LangChain 的 InMemoryChatMessageHistory 会话。"""
    history = InMemoryChatMessageHistory()
    store[session_id] = history
    for record in records:
        role = record.get("role")
        content = record.get("content", "")
        if role == "human":
            history.add_user_message(content)
        elif role == "ai":
            history.add_ai_message(content)


def build_chain(llm: Any, profile_text: str = "") -> RunnableWithMessageHistory:
    """组装 LangChain 可运行链：注入 emotion_context → prompt → LLM，并挂载消息历史。"""
    system_message = build_system_message(profile_text)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = (
        RunnablePassthrough.assign(
            emotion_context=lambda input_data: input_data.get("emotion_context", "")
        )
        | prompt
        | llm
    )
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
