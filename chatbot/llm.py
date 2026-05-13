from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from chatbot.config import ChatConfig


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


def build_chain(llm: ChatOpenAI) -> LLMChain:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return LLMChain(llm=llm, prompt=prompt, memory=memory)
