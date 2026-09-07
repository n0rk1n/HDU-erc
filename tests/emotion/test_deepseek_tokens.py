import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from chatbot.core.errors import ConfigError
from chatbot.emotion.config import load_emotion_settings
from chatbot.emotion.model import ModelTokenCounter


def test_deepseek_counter_counts_full_official_prompt():
    counter = ModelTokenCounter(object(), tokenizer_model='deepseek-v4-flash')
    from tokenizers import Tokenizer
    from chatbot.llm.deepseek_tokens import TOKENIZER_PATH
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    messages = [SystemMessage(content='你是助手。'), HumanMessage(content='你好'),
                AIMessage(content='你好！'), HumanMessage(content='今天很开心😊')]
    # Official V4 text-chat format includes BOS, roles, EOS and generation prefix.
    prompt = ('<｜begin▁of▁sentence｜>你是助手。<｜User｜>你好'
              '<｜Assistant｜></think>你好！<｜end▁of▁sentence｜>'
              '<｜User｜>今天很开心😊<｜Assistant｜><think>')
    assert counter.count(messages) == len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    assert counter.count([]) == 0
    assert '60d8d707' in counter.version


@pytest.mark.parametrize('message', [
    ToolMessage(content='tool result', tool_call_id='1'),
    HumanMessage(content=[{'type': 'image_url', 'image_url': {'url': 'https://example.org/a.png'}}]),
    AIMessage(content='', tool_calls=[{'name': 'f', 'args': {}, 'id': '1'}]),
])
def test_deepseek_rejects_unaccounted_message_payload(message):
    counter = ModelTokenCounter(object(), tokenizer_model='deepseek-v4-flash')
    with pytest.raises(ConfigError, match='text-only'):
        counter.count([message])


def test_deepseek_defaults_preserve_explicit_budget(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_MODEL', 'deepseek-v4-flash')
    monkeypatch.delenv('EMOTION_LLM_MODEL', raising=False)
    monkeypatch.setenv('EMOTION_CONTEXT_TOKENS', '')
    monkeypatch.setenv('EMOTION_TOKENIZER_MODEL', '')
    settings = load_emotion_settings(None)
    assert settings.budget.context_tokens == 1048576
    assert settings.tokenizer_model == 'deepseek-v4-flash'
    monkeypatch.setenv('EMOTION_CONTEXT_TOKENS', '32000')
    assert load_emotion_settings(None).budget.context_tokens == 32000


def test_role_override_does_not_inherit_deepseek_defaults(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'deepseek-v4-flash')
    monkeypatch.setenv('EMOTION_LLM_MODEL', 'other-model')
    monkeypatch.setenv('EMOTION_CONTEXT_TOKENS', '')
    monkeypatch.setenv('EMOTION_TOKENIZER_MODEL', '')
    with pytest.raises(ConfigError, match='EMOTION_CONTEXT_TOKENS'):
        load_emotion_settings(None)


def test_startup_with_deepseek_and_blank_budget(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    from dotenv import dotenv_values
    from fastapi.testclient import TestClient
    from chatbot.web import create_app

    # Fresh example configuration, no developer secrets and no model API calls.
    for key in list(os.environ):
        if key.startswith(('LLM_', 'CHAT_LLM_', 'EMOTION_')):
            monkeypatch.delenv(key)
    root = Path(__file__).resolve().parents[2]
    for key, value in dotenv_values(root / '.env.example').items():
        monkeypatch.setenv(key, value or '')
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.setenv('LLM_MODEL', 'deepseek-v4-flash')
    monkeypatch.setenv('LLM_BASE_URL', 'https://api.deepseek.com')
    monkeypatch.setenv('SQLITE_DB_PATH', str(tmp_path / 'startup.sqlite3'))
    with TestClient(create_app()) as client:
        assert client.get('/').status_code == 200
