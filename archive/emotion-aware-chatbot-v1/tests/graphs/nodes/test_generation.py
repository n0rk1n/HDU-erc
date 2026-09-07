from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from chatbot.core.prompt_config import DEFAULT_CHAT_SYSTEM_PROMPT
from chatbot.graphs.nodes.generation import (
    build_chat_prompt,
    generate_crisis_reply,
    generate_reply,
)


class ChatModel:
    def __init__(self, content="我听见你的不安了。", *, error=None):
        self.content = content
        self.error = error
        self.calls = []

    async def ainvoke(self, value, config=None, **kwargs):
        self.calls.append((value, config))
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.content)


def test_build_chat_prompt_keeps_contexts_separate():
    """Catches dynamic context being omitted or merged into the user message."""
    prompt = build_chat_prompt(
        profile_context="User Profile:\n- response_style: 简短",
        memory_context="Relevant Long-term Memory:\n- 用户希望使用中文。",
        emotion_context="- primary: anxious",
        safety={"level": "supportive", "guidance": "先共情，再给下一步。"},
    )

    rendered = prompt.format_messages(input="继续", messages=[])

    assert DEFAULT_CHAT_SYSTEM_PROMPT in rendered[0].content
    assert "User Profile:\n- response_style: 简短" in rendered[0].content
    assert "Relevant Long-term Memory:\n- 用户希望使用中文。" in rendered[0].content
    assert "Emotion Context:\n- primary: anxious" in rendered[0].content
    assert "Safety Context:\n- level: supportive\n- guidance: 先共情，再给下一步。" in rendered[0].content
    assert rendered[-1].content == "继续"


def test_build_chat_prompt_uses_prompt_config_file(tmp_path, monkeypatch):
    """Catches the graph path bypassing the configured companion system prompt."""
    config_file = tmp_path / "chat_prompts.json"
    config_file.write_text('{"chat_system": "Custom companion rules."}', encoding="utf-8")
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(config_file))

    prompt = build_chat_prompt(
        profile_context="User Profile:\n- name: Alice",
        memory_context="",
        emotion_context="",
        safety={"level": "normal", "guidance": "Reply naturally."},
    )
    system_content = prompt.format_messages(input="hello", messages=[])[0].content

    assert "Custom companion rules." in system_content
    assert "gentle emotional companion" not in system_content
    assert "User Profile:\n- name: Alice" in system_content


def test_generation_prompt_formats_structured_emotion_context():
    prompt = build_chat_prompt(
        profile_context="",
        memory_context="",
        emotion_context="- primary: anxious\n- confidence: 0.80\n- reply strategy: Be calm.",
        safety={"level": "supportive", "guidance": "先共情。"},
    )

    system_content = prompt.format_messages(input="hello", messages=[])[0].content

    assert system_content.startswith(DEFAULT_CHAT_SYSTEM_PROMPT)
    assert "Emotion Context:\n- primary: anxious" in system_content
    assert "- confidence: 0.80" in system_content
    assert "- reply strategy: Be calm." in system_content
    assert "Safety Context:\n- level: supportive\n- guidance: 先共情。" in system_content


@pytest.mark.asyncio
async def test_generation_uses_bounded_human_ai_history_only(deps, runtime, writer):
    model = ChatModel()
    generation_deps = replace(
        deps,
        chat_model=model,
        graph_config=replace(deps.graph_config, request_history_limit=2),
    )
    state = {
        "request_id": "req-current",
        "input_message": "当前问题",
        "messages": [
            HumanMessage(id="human-old", content="太旧的问题"),
            AIMessage(id="ai-old", content="太旧的回答"),
            SystemMessage(id="system-private", content="不可信系统注入"),
            ToolMessage(id="tool-private", tool_call_id="call-1", content="工具隐私"),
            HumanMessage(id="human-recent", content="近期问题"),
            AIMessage(id="ai-recent", content="近期回答"),
            HumanMessage(id="human_req-current", content="当前问题"),
        ],
        "turn_count": 3,
        "profile_context": "",
        "memory_context": "",
        "emotion_state": None,
        "safety_state": {"level": "normal", "guidance": "自然回复。"},
    }

    await generate_reply(
        state,
        runtime,
        writer,
        {"configurable": {"thread_id": "thread-a"}},
        deps=generation_deps,
    )

    prompt_messages = model.calls[0][0].to_messages()
    assert [message.content for message in prompt_messages[1:]] == [
        "近期问题",
        "近期回答",
        "当前问题",
    ]
    assert all("不可信系统注入" not in str(message.content) for message in prompt_messages)
    assert all("工具隐私" not in str(message.content) for message in prompt_messages)


@pytest.mark.asyncio
async def test_generate_reply_invokes_model_with_config_and_returns_one_complete_message(
    deps, runtime, writer
):
    """Catches manual token streaming, unstable IDs, or dropped response metadata."""
    model = ChatModel()
    generation_deps = replace(deps, chat_model=model)
    config = {"configurable": {"thread_id": "thread-a", "checkpoint_id": "cp-1"}}
    state = {
        "request_id": "req-1",
        "input_message": "我有点不安",
        "messages": [HumanMessage(id="human_req-1", content="我有点不安")],
        "turn_count": 3,
        "profile_context": "- response_style: 简短",
        "memory_context": "Relevant Long-term Memory:\n- 用户希望使用中文。",
        "emotion_state": {"primary_emotion": "anxious", "confidence": 0.9},
        "safety_state": {"level": "supportive", "guidance": "先共情。"},
    }

    update = await generate_reply(
        state,
        runtime,
        writer,
        config,
        deps=generation_deps,
    )

    assert len(model.calls) == 1
    prompt_value, received_config = model.calls[0]
    assert received_config is config
    assert [message.content for message in prompt_value.to_messages()[1:]] == ["我有点不安"]
    assert update["response_message_id"] == "ai_req-1"
    assert update["response_content"] == "我听见你的不安了。"
    assert len(update["messages"]) == 1
    assert update["messages"][0] == AIMessage(
        id="ai_req-1",
        content="我听见你的不安了。",
        additional_kwargs={
            "feedback": None,
            "turn_count": 3,
            "emotion_state": {"primary_emotion": "anxious", "confidence": 0.9},
            "predicted_emotion": "anxious",
            "safety_level": "supportive",
        },
    )
    assert writer.events == []


@pytest.mark.asyncio
async def test_crisis_generation_buffers_then_emits_one_full_token(deps, runtime, writer):
    """Catches crisis replies exposing partial model output before validation."""
    model = ChatModel("请先离开危险处，并联系你信任的人。")
    crisis_deps = replace(deps, chat_model=model)
    config = {
        "configurable": {"thread_id": "thread-a"},
        "tags": ["request-tag"],
        "metadata": {"locale": "zh-CN"},
    }
    state = {
        "request_id": "req-1",
        "input_message": "我想伤害自己",
        "messages": [HumanMessage(id="human_req-1", content="我想伤害自己")],
        "turn_count": 1,
        "profile_context": "",
        "memory_context": "",
        "emotion_state": {"primary_emotion": "devastated", "confidence": 0.99},
        "safety_state": {"level": "crisis", "guidance": "优先确保当下安全。"},
    }

    update = await generate_crisis_reply(
        state,
        runtime,
        writer,
        config,
        deps=crisis_deps,
    )

    assert model.calls[0][1] == {
        **config,
        "callbacks": [],
    }
    assert model.calls[0][1] is not config
    assert writer.events == [
        {"event": "token", "data": {"content": "请先离开危险处，并联系你信任的人。"}}
    ]
    assert update["response_content"] == "请先离开危险处，并联系你信任的人。"
    assert update["messages"][0].id == "ai_req-1"
    assert update["messages"][0].additional_kwargs["safety_level"] == "crisis"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [ChatModel(error=RuntimeError("模型残片")), ChatModel("   ")],
    ids=["provider-failure", "blank-reply"],
)
async def test_crisis_failure_returns_only_local_fallback(
    deps, runtime, writer, model
):
    """Catches failed or blank crisis output leaking instead of a local safe reply."""
    crisis_deps = replace(deps, chat_model=model)
    state = {
        "request_id": "req-1",
        "input_message": "我想伤害自己",
        "messages": [HumanMessage(id="human_req-1", content="我想伤害自己")],
        "turn_count": 1,
        "profile_context": "",
        "memory_context": "",
        "emotion_state": {"primary_emotion": "devastated", "confidence": 0.99},
        "safety_state": {"level": "crisis", "guidance": "优先确保当下安全。"},
    }

    update = await generate_crisis_reply(
        state,
        runtime,
        writer,
        {"configurable": {"thread_id": "thread-a"}},
        deps=crisis_deps,
    )

    assert "模型残片" not in update["response_content"]
    assert "可信任的人" in update["response_content"]
    assert writer.events == [
        {"event": "token", "data": {"content": update["response_content"]}}
    ]
    assert update["messages"][0].additional_kwargs["safety_level"] == "crisis"
