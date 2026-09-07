import json
from types import SimpleNamespace

import pytest

from chatbot.core.errors import ConfigError
from chatbot.emotion.config import CONFIG_ROOT, load_taxonomy
from chatbot.emotion.prompt import load_examples, prepare_analysis
from chatbot.emotion.types import BudgetConfig
from tests.emotion.test_budget import UnitCounter


def prepare():
    taxonomy = load_taxonomy()
    row = SimpleNamespace(id='u', role='user', status='completed', content='你好',
                          conversation_id='c', request_id='r', sequence_no=1)
    return prepare_analysis([row], current_id='u', taxonomy=taxonomy,
                            examples=load_examples(CONFIG_ROOT / 'emotion_examples.json', taxonomy),
                            recent_labels=[], counter=UnitCounter(),
                            budget=BudgetConfig(100000, 1000, 256))


def test_config_changes_reach_prompt_and_audit_next_analysis(tmp_path, monkeypatch):
    path = tmp_path / 'prompts.json'
    monkeypatch.setenv('EMOTION_SYSTEM_PROMPT_PATH', str(path))
    path.write_text(json.dumps({'version': 'custom-1', 'emotion_system': '自定义识别规则'}))
    first = prepare()
    assert first.messages[0].content.startswith('自定义识别规则\n')
    assert '标签及描述：' in first.messages[0].content
    assert '参考示例：' in first.messages[0].content
    assert first.messages[-1].content == '你好'
    assert first.snapshot['prompt_version'] == 'custom-1'
    assert first.snapshot['prompt'][0]['content'] == first.messages[0].content
    path.write_text(json.dumps({'version': 'custom-1', 'emotion_system': '更新识别规则'}))
    second = prepare()
    assert second.messages[0].content.startswith('更新识别规则\n')
    assert second.snapshot['prompt_config_hash'] != first.snapshot['prompt_config_hash']


@pytest.mark.parametrize('content', [None, '', '{', '[]', '{}',
    '{"version":"v","emotion_system":" "}',
    '{"version":2,"emotion_system":"rules"}',
    '{"version":" ","emotion_system":"rules"}',
    '{"version":"v","emotion_system":[]}',
    '{"version":"v","version":"other","emotion_system":"rules"}'])
def test_invalid_config_is_not_silently_replaced(tmp_path, monkeypatch, content):
    path = tmp_path / 'prompts.json'
    if content is not None:
        path.write_text(content)
    monkeypatch.setenv('EMOTION_SYSTEM_PROMPT_PATH', str(path))
    with pytest.raises(ConfigError):
        prepare()


@pytest.mark.parametrize('setting', [None, '', '   '])
def test_default_config_is_resolved_independently_of_cwd(tmp_path, monkeypatch, setting):
    from chatbot.emotion.prompt import get_emotion_prompt
    if setting is None:
        monkeypatch.delenv('EMOTION_SYSTEM_PROMPT_PATH', raising=False)
    else:
        monkeypatch.setenv('EMOTION_SYSTEM_PROMPT_PATH', setting)
    monkeypatch.chdir(tmp_path)
    config = get_emotion_prompt()
    assert config['emotion_system'].strip()
    assert config['version'].strip()
