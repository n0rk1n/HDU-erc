import json
import pytest
from chatbot.emotion.config import load_taxonomy

def payload(**changes):
    return dict(primary_emotion='no_emotion',confidence=.9,secondary_emotions=[],evidence='转换文件',reply_strategy='直接处理',trajectory_note='',safety_level='normal',**changes)

def test_no_emotion_is_success_and_family_derived():
    from chatbot.emotion.parsing import parse_result
    result=parse_result(json.dumps(payload()),load_taxonomy())
    assert result.primary_emotion=='no_emotion'
    assert result.primary_family=='no_emotion'

@pytest.mark.parametrize('field,value',[('secondary_emotions',['sad']),('primary_emotion','unknown'),('confidence',float('nan')),('confidence',True),('safety_level','invalid')])
def test_invalid_result_is_not_normalized_into_success(field,value):
    from chatbot.emotion.parsing import parse_result
    obj=payload();obj[field]=value
    with pytest.raises(ValueError):
        parse_result(json.dumps(obj),load_taxonomy())
