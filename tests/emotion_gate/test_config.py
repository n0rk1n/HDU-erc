import json
from dataclasses import replace
import pytest
from chatbot.core.errors import ConfigError


def test_interval_boundaries():
    from chatbot.emotion_gate.types import GatePolicy
    from chatbot.emotion_gate.policy import forced_reason
    for current, last, expected in [(1,None,'first_turn'),(15,1,None),(16,1,'interval_reached'),(22,8,None),(23,8,'interval_reached')]:
        assert forced_reason(GatePolicy(),current_turn=current,last_success_turn=last,latest_analysis_failed=False)==expected
    for first in (True,False):
        p=GatePolicy(force_first_analysis=first,retry_failed_analysis_next_turn=False,max_interval_turns=2)
        assert forced_reason(p,current_turn=2,last_success_turn=None,latest_analysis_failed=True)=='interval_reached'
    assert forced_reason(GatePolicy(),current_turn=3,last_success_turn=1,latest_analysis_failed=True)=='previous_analysis_failed'

@pytest.mark.parametrize('overrides',[{'max_interval_turns':0},{'history_turn_limit':-1},{'max_interval_turns':True},{'gate_failure_action':'ignore'},{'force_first_analysis':1},{'retry_delay_seconds':float('inf')},{'model_retry_count':-1}])
def test_invalid_policy(overrides):
    from chatbot.emotion_gate.types import GatePolicy
    with pytest.raises(ValueError): GatePolicy(**overrides)


def test_settings_prompt_override_and_budget(tmp_path,monkeypatch):
    from chatbot.emotion.config import EmotionSettings
    from chatbot.emotion.types import BudgetConfig
    from chatbot.emotion_gate.config import load_gate_settings
    from pydantic import SecretStr
    emotion=EmotionSettings(SecretStr('test'), 'fake',None,0,3,BudgetConfig(10000,100,10),'fake',tmp_path,tmp_path,tmp_path)
    prompt=tmp_path/'prompt.json';prompt.write_text(json.dumps({'version':'custom','system':'custom decision instruction'}))
    config=tmp_path/'gate.json';config.write_text(json.dumps({'version':'v1','history_turn_limit':9,'max_interval_turns':2,'history_ratio':0.3}))
    monkeypatch.setenv('EMOTION_GATE_CONFIG_PATH',str(config));monkeypatch.setenv('EMOTION_GATE_SYSTEM_PROMPT_PATH',str(prompt))
    result=load_gate_settings(emotion)
    assert result.policy.history_turn_limit==9 and result.policy.max_interval_turns==2
    assert result.prompt['system']=='custom decision instruction'
    assert result.budget.history_ratio==0.3
    config.write_text('{"version":"v1","unknown":1}')
    with pytest.raises(ConfigError):load_gate_settings(emotion)
    config.write_text('{"version":"v1","history_turn_limit":2,"history_turn_limit":3}')
    with pytest.raises(ConfigError):load_gate_settings(emotion)

@pytest.mark.parametrize('tokenizer',['',False,1,[]])
def test_explicit_invalid_tokenizer_is_rejected(tmp_path,monkeypatch,tokenizer):
    from chatbot.emotion.config import EmotionSettings
    from chatbot.emotion.types import BudgetConfig
    from chatbot.emotion_gate.config import load_gate_settings
    from pydantic import SecretStr
    emotion=EmotionSettings(SecretStr('test'),'fake',None,0,3,BudgetConfig(10000,100,10),'fake',tmp_path,tmp_path,tmp_path)
    config=tmp_path/'invalid.json'
    config.write_text(json.dumps({'version':'v1','tokenizer_model':tokenizer}))
    monkeypatch.setenv('EMOTION_GATE_CONFIG_PATH',str(config))
    with pytest.raises(ConfigError):load_gate_settings(emotion)
