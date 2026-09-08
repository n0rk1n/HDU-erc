import json
import pytest
from chatbot.emotion.config import load_taxonomy

def payload(**changes):
    return dict(primary_emotion='neutral',confidence=.9,secondary_emotions=[],evidence='转换文件',reply_strategy='直接处理',trajectory_note='',safety_level='normal') | changes

def test_neutral_is_success_and_family_derived():
    from chatbot.emotion.parsing import parse_result
    result=parse_result(json.dumps(payload()),load_taxonomy())
    assert result.primary_emotion=='neutral'
    assert result.primary_family=='neutral'


@pytest.mark.parametrize('primary,secondary', [
    ('anger', ['annoyance']), ('neutral', ['sadness']), ('sadness', ['neutral']),
])
def test_goemotions_cooccurring_labels_are_preserved(primary, secondary):
    from chatbot.emotion.parsing import parse_result
    result = parse_result(json.dumps(payload(primary_emotion=primary, secondary_emotions=secondary)), load_taxonomy())
    assert result.primary_emotion == primary
    assert result.secondary_emotions == secondary
    assert result.primary_family == load_taxonomy().families[primary]

@pytest.mark.parametrize('field,value',[
    ('secondary_emotions',['sad']), ('secondary_emotions',['no_emotion']),
    ('secondary_emotions',['neutral']), ('secondary_emotions',['anger','anger']),
    ('primary_emotion','unknown'), ('primary_emotion','no_emotion'), ('primary_emotion','sad'),
    ('confidence',float('nan')),('confidence',True),('safety_level','invalid')])
def test_invalid_result_is_not_normalized_into_success(field,value):
    from chatbot.emotion.parsing import parse_result
    obj=payload();obj[field]=value
    with pytest.raises(ValueError):
        parse_result(json.dumps(obj),load_taxonomy())
