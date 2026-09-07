"""Protect full GLM prompt accounting and startup without an OpenAI fallback."""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from tokenizers import Tokenizer

from chatbot.core.errors import ConfigError
from chatbot.emotion.model import ModelTokenCounter


def test_glm53_counts_roles_effort_and_generation_prefix():
    counter = ModelTokenCounter(object(), tokenizer_model='ZHIPU/GLM-5.3')
    tokenizer = Tokenizer.from_file(str(Path(__file__).resolve().parents[2]
                                        / 'chatbot/llm/vendor/glm53/tokenizer.json'))
    messages = [SystemMessage(content='你是助手。'), HumanMessage(content='你好'),
                AIMessage(content='  你好！  '), HumanMessage(content='今天很开心😊')]
    # Hand-rendered from the pinned official text-only template.
    prompts = [f'[gMASK]<sop><|system|>Reasoning Effort: {effort}'
               '<|system|>你是助手。<|user|>你好<|assistant|><think></think>你好！'
               '<|user|>今天很开心😊<|assistant|><think>'
               for effort in ('Low', 'High', 'Max')]
    assert counter.count(messages) == max(len(tokenizer.encode(p, add_special_tokens=False).ids)
                                           for p in prompts)
    assert counter.count([]) == 0
    assert counter.identity == 'ZHIPU/GLM-5.3'


def test_glm53_includes_preserved_assistant_reasoning():
    counter = ModelTokenCounter(object(), tokenizer_model='ZHIPU/GLM-5.3')
    plain = [HumanMessage(content='你好'), AIMessage(content='你好！'), HumanMessage(content='继续')]
    reasoned = [plain[0], AIMessage(content='你好！', additional_kwargs={
        'reasoning_content': '先分析对方的语气，再给出简短回复。' * 10}), plain[2]]
    assert counter.count(reasoned) > counter.count(plain)
    embedded = [plain[0], AIMessage(content='<think>先分析对方的语气，再给出简短回复。</think>你好！'), plain[2]]
    explicit = [plain[0], AIMessage(content='你好！', additional_kwargs={
        'reasoning_content': '先分析对方的语气，再给出简短回复。'}), plain[2]]
    assert counter.count(embedded) == counter.count(explicit)


@pytest.mark.parametrize('message', [
    ToolMessage(content='result', tool_call_id='1'),
    HumanMessage(content=[{'type': 'image_url', 'image_url': {'url': 'https://example.org/a.png'}}]),
    AIMessage(content='', tool_calls=[{'name': 'f', 'args': {}, 'id': '1'}]),
    HumanMessage(content='你好', additional_kwargs={'unknown': 'payload'}),
])
def test_glm53_rejects_unaccounted_payloads(message):
    counter = ModelTokenCounter(object(), tokenizer_model='ZHIPU/GLM-5.3')
    with pytest.raises(ConfigError, match='text-only'):
        counter.count([message])


def test_startup_with_glm53_and_matching_budget(monkeypatch, tmp_path):
    import os
    from dotenv import dotenv_values
    from fastapi.testclient import TestClient
    from chatbot.web import create_app

    for key in list(os.environ):
        if key.startswith(('LLM_', 'CHAT_LLM_', 'EMOTION_')):
            monkeypatch.delenv(key)
    root = Path(__file__).resolve().parents[2]
    for key, value in dotenv_values(root / '.env.example').items():
        monkeypatch.setenv(key, value or '')
    for key, value in {
        'PYTHON_DOTENV_DISABLED': '1', 'LLM_MODEL': 'ZHIPU/GLM-5.3',
        'LLM_BASE_URL': 'https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
        'LLM_THINKING': 'enabled', 'LLM_REASONING_EFFORT': 'low',
        'EMOTION_CONTEXT_TOKENS': '1048576', 'EMOTION_TOKENIZER_MODEL': 'ZHIPU/GLM-5.3',
        'SQLITE_DB_PATH': str(tmp_path / 'startup.sqlite3'),
    }.items():
        monkeypatch.setenv(key, value)
    with TestClient(create_app()) as client:
        assert client.get('/').status_code == 200
