"""Catch cross-role inheritance and ignored role-specific connection overrides."""
import json
import os

import pytest

from chatbot.core.config import AppConfig
from chatbot.core.errors import ConfigError
from chatbot.emotion.config import load_emotion_settings
from chatbot.emotion_gate.config import load_gate_settings


@pytest.fixture
def model_env(monkeypatch, tmp_path):
    # Use an isolated dotenv path so developer credentials never affect these tests.
    monkeypatch.setattr('chatbot.core.config.PROJECT_ROOT', tmp_path)
    for name in list(os.environ):
        if name.startswith(('LLM_', 'CHAT_LLM_', 'EMOTION_', 'CHAT_SYSTEM_')):
            monkeypatch.delenv(name)
    for name, value in {
        'LLM_API_KEY': 'default-key', 'LLM_MODEL': 'default-model',
        'LLM_BASE_URL': 'https://default.example/v1',
        'LLM_TEMPERATURE': '0.4', 'LLM_TIMEOUT_SECONDS': '17',
        'EMOTION_CONTEXT_TOKENS': '10000', 'EMOTION_TOKENIZER_MODEL': 'default-tokenizer',
    }.items():
        monkeypatch.setenv(name, value)
    policy = tmp_path / 'gate.json'
    policy.write_text(json.dumps({'version': 'v1', 'context_tokens': 10000,
                                  'tokenizer_model': 'gate-tokenizer'}))
    monkeypatch.setenv('EMOTION_GATE_CONFIG_PATH', str(policy))
    return monkeypatch


def settings():
    chat = AppConfig.from_env()
    emotion = load_emotion_settings(chat)
    gate = load_gate_settings(emotion)
    return [
        (chat.llm_api_key.get_secret_value(), chat.llm_model, chat.llm_base_url,
         chat.llm_temperature, chat.llm_timeout_seconds),
        (emotion.api_key.get_secret_value(), emotion.model, emotion.base_url,
         emotion.temperature, emotion.timeout_seconds),
        (gate.api_key.get_secret_value(), gate.model, gate.base_url,
         gate.temperature, gate.timeout_seconds),
    ]


DEFAULT = ('default-key', 'default-model', 'https://default.example/v1', 0.4, 17)


@pytest.mark.parametrize('value,expected', [
    (None, 'disabled'), ('', 'disabled'), ('   ', 'disabled'),
    ('disabled', 'disabled'), ('enabled', 'enabled'), (' enabled ', 'enabled'),
])
def test_chat_thinking_environment(model_env, value, expected):
    """Catches ignored thinking overrides and a changed default for replies."""
    if value is not None:
        model_env.setenv('CHAT_LLM_THINKING', value)
    assert AppConfig.from_env().llm_thinking == expected


@pytest.mark.parametrize('value', ['true', 'false', 'enable', 'invalid'])
def test_chat_thinking_rejects_invalid_values(model_env, value):
    """Catches configuration typos silently restoring provider thinking defaults."""
    model_env.setenv('CHAT_LLM_THINKING', value)
    with pytest.raises(ConfigError, match='CHAT_LLM_THINKING'):
        AppConfig.from_env()


def thinking_settings():
    chat = AppConfig.from_env()
    emotion = load_emotion_settings(chat)
    gate = load_gate_settings(emotion)
    return chat, emotion, gate


@pytest.mark.parametrize('default', ['disabled', 'enabled'])
@pytest.mark.parametrize('blank', [None, '', '   '])
def test_thinking_roles_inherit_shared_default(model_env, default, blank):
    """Catches a role retaining its own default when the shared setting changes."""
    model_env.setenv('LLM_THINKING', default)
    if blank is not None:
        for prefix in ('CHAT_LLM', 'EMOTION_LLM', 'EMOTION_GATE_LLM'):
            model_env.setenv(f'{prefix}_THINKING', blank)
    chat, emotion, gate = thinking_settings()
    assert [chat.llm_thinking, emotion.thinking, gate.thinking] == [default] * 3


@pytest.mark.parametrize('role,prefix', list(enumerate(('CHAT_LLM', 'EMOTION_LLM', 'EMOTION_GATE_LLM'))))
@pytest.mark.parametrize('default,override', [('disabled', 'enabled'), ('enabled', 'disabled')])
def test_thinking_override_reaches_only_its_role_request_and_audit(model_env, role, prefix, default, override):
    """Catches ignored overrides, cross-role leakage, or request/audit divergence."""
    from chatbot.emotion.model import OpenAICompatibleEmotionModel
    from chatbot.llm.openai_compatible import OpenAICompatibleChatModel
    from langchain_core.messages import HumanMessage
    model_env.setenv('LLM_MODEL', 'deepseek-v4-flash')
    model_env.setenv('LLM_BASE_URL', 'https://api.deepseek.com')
    model_env.setenv('LLM_THINKING', default)
    model_env.setenv(f'{prefix}_THINKING', override)
    chat, emotion, gate = thinking_settings()
    adapters = [OpenAICompatibleChatModel(chat), OpenAICompatibleEmotionModel(emotion),
                OpenAICompatibleEmotionModel(gate)]
    expected = [default] * 3
    expected[role] = override
    for adapter, mode in zip(adapters, expected):
        payload = adapter.client._get_request_payload([HumanMessage(content='你好')])
        assert payload['extra_body'] == {'thinking': {'type': mode}}
        assert adapter.parameters['extra_body'] == {'thinking': {'type': mode}}


@pytest.mark.parametrize('prefix', ['LLM', 'CHAT_LLM', 'EMOTION_LLM', 'EMOTION_GATE_LLM'])
def test_invalid_thinking_is_rejected_for_each_role(model_env, prefix):
    model_env.setenv(f'{prefix}_THINKING', 'invalid')
    with pytest.raises(ConfigError, match=f'{prefix}_THINKING'):
        thinking_settings()


@pytest.mark.parametrize('blank', [None, '', '   '])
def test_all_roles_use_shared_defaults(model_env, blank):
    if blank is not None:
        for prefix in ('CHAT_LLM', 'EMOTION_LLM', 'EMOTION_GATE_LLM'):
            for field in ('API_KEY', 'MODEL', 'BASE_URL', 'TEMPERATURE', 'TIMEOUT_SECONDS'):
                model_env.setenv(f'{prefix}_{field}', blank)
    assert settings() == [DEFAULT, DEFAULT, DEFAULT]


@pytest.mark.parametrize('role,prefix', list(enumerate(('CHAT_LLM', 'EMOTION_LLM', 'EMOTION_GATE_LLM'))))
def test_role_override_does_not_leak_to_other_roles(model_env, role, prefix):
    for field, value in {'API_KEY': 'role-key', 'MODEL': 'role-model',
                         'BASE_URL': 'https://role.example/v1', 'TEMPERATURE': '0',
                         'TIMEOUT_SECONDS': '8'}.items():
        model_env.setenv(f'{prefix}_{field}', value)
    expected = [DEFAULT, DEFAULT, DEFAULT]
    expected[role] = ('role-key', 'role-model', 'https://role.example/v1', 0, 8)
    assert settings() == expected


def test_all_roles_can_select_models_independently(model_env):
    for prefix, model in [('CHAT_LLM', 'chat'), ('EMOTION_LLM', 'emotion'), ('EMOTION_GATE_LLM', 'gate')]:
        model_env.setenv(f'{prefix}_MODEL', model)
    assert settings() == [
        ('default-key', 'chat', 'https://default.example/v1', 0.4, 17),
        ('default-key', 'emotion', 'https://default.example/v1', 0.4, 17),
        ('default-key', 'gate', 'https://default.example/v1', 0.4, 17),
    ]


@pytest.mark.parametrize('prefix', ['LLM', 'CHAT_LLM', 'EMOTION_LLM', 'EMOTION_GATE_LLM'])
@pytest.mark.parametrize('field,value', [('TEMPERATURE', 'nan'), ('TEMPERATURE', '3'),
                                        ('TIMEOUT_SECONDS', 'inf'), ('TIMEOUT_SECONDS', '0'),
                                        ('TIMEOUT_SECONDS', 'invalid')])
def test_invalid_effective_connection_is_rejected(model_env, prefix, field, value):
    model_env.setenv(f'{prefix}_{field}', value)
    with pytest.raises(ConfigError):
        settings()


def test_gate_default_differs_from_emotion_requires_matching_budget(model_env, tmp_path):
    model_env.setenv('EMOTION_LLM_MODEL', 'emotion-only-model')
    policy = tmp_path / 'gate.json'
    policy.write_text(json.dumps({'version': 'v1'}))
    with pytest.raises(ConfigError, match='different gate model'):
        settings()


def test_matching_default_model_can_inherit_emotion_budget(model_env, tmp_path):
    policy = tmp_path / 'gate.json'
    policy.write_text(json.dumps({'version': 'v1'}))
    chat = AppConfig.from_env()
    emotion = load_emotion_settings(chat)
    gate = load_gate_settings(emotion)
    assert gate.model == 'default-model'
    assert gate.tokenizer_model == 'default-tokenizer'
    assert gate.budget.context_tokens == 10000
