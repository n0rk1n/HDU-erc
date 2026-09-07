from dataclasses import replace
import pytest
from chatbot.db.messages import MessageRepository
from chatbot.services.identity import IdentityService
from tests.emotion_gate.helpers import complete
from tests.emotion.helpers import Counter

@pytest.mark.parametrize('limit,want',[ (0,['current']), (1,['r6','reply','current']), (5,['r2','reply','r3','reply','r4','reply','r5','reply','r6','reply','current']), (9,['r0','reply','r1','reply','r2','reply','r3','reply','r4','reply','r5','reply','r6','reply','current'])])
async def test_whole_turn_window(database,limit,want):
    from tests.emotion_gate.helpers import gate_runtime
    from chatbot.emotion_gate.prompt import prepare_gate_prompt
    from chatbot.emotion_gate.types import GatePolicy
    messages=MessageRepository(database);convo=(await IdentityService(database).resolve('window')).conversation
    for i in range(7):
        turn=await messages.reserve_turn(convo.id,str(i),f'r{i}');await complete(messages,turn)
    current=await messages.reserve_turn(convo.id,'now','current')
    rows=await messages.list_emotion_history(convo.id,through_sequence=current.user.sequence_no)
    settings=gate_runtime(database,policy=GatePolicy(history_turn_limit=limit)).settings
    settings=replace(settings,prompt={'version':'custom','system':'my custom gate'})
    result=prepare_gate_prompt(rows,current_id=current.user.id,baseline=None,settings=settings,counter=Counter())
    assert [m.content for m in result.messages[1:]]==want
    assert result.messages[0].content.startswith('my custom gate')
    assert result.snapshot['prompt_version']=='custom'

@pytest.mark.parametrize('raw',['{}','{"should_analyze":"false","reason":"ok"}','{"should_analyze":true,"reason":""}','bad','{"should_analyze":true,"should_analyze":false,"reason":"x"}'])
def test_invalid_output(raw):
    from chatbot.emotion_gate.parsing import parse_gate_result
    with pytest.raises(ValueError):parse_gate_result(raw)

def test_valid_output():
    from chatbot.emotion_gate.parsing import parse_gate_result
    assert parse_gate_result('{"should_analyze":false,"reason":"stable"}').should_analyze is False
