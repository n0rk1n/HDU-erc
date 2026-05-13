import warnings

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory is deprecated.*")

from chatbot.config import ChatConfig

store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


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


def init_session_history(session_id: str, records: list[dict]) -> None:
    history = get_session_history(session_id)
    for record in records:
        role = record.get("role")
        content = record.get("content", "")
        if role == "human":
            history.add_user_message(content)
        elif role == "ai":
            history.add_ai_message(content)


def build_chain(llm: ChatOpenAI, profile_text: str = "") -> RunnableWithMessageHistory:
    system_message = "You are a helpful assistant."
    if profile_text:
        system_message += f"\n\nUser Profile:\n{profile_text}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
