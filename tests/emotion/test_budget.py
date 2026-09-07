import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class UnitCounter:
    identity='unit'
    version='1'
    def count(self,messages):
        return sum(len(m.content)+1 for m in messages)

def test_keeps_latest_complete_pairs():
    from chatbot.emotion.budget import select_history
    from chatbot.emotion.history import DialogueTurn
    from chatbot.emotion.types import BudgetConfig
    turns=[DialogueTurn(str(i),(HumanMessage(content='aaa',id=f'u{i}'),AIMessage(content='bbb',id=f'a{i}')),(f'u{i}',f'a{i}')) for i in range(3)]
    selected=select_history(turns,HumanMessage(content='now',id='current'),counter=UnitCounter(),budget=BudgetConfig(40,8,2),system_messages=[])
    assert [m.id for m in selected.messages]==['u1','a1','u2','a2','current']
    assert selected.audit['chat_tokens']==20

@pytest.mark.parametrize('system,current', [('', 'x'*24),('x'*30,'now')])
def test_never_truncates_current_to_fit(system,current):
    from chatbot.emotion.budget import select_history, ContextBudgetExceeded
    from chatbot.emotion.types import BudgetConfig
    with pytest.raises(ContextBudgetExceeded):
        select_history([],HumanMessage(content=current,id='u'),counter=UnitCounter(),budget=BudgetConfig(40,8,2),system_messages=[SystemMessage(content=system)] if system else [])

def test_long_recent_pair_does_not_allow_older_short_pairs():
    from chatbot.emotion.budget import select_history
    from chatbot.emotion.history import DialogueTurn
    from chatbot.emotion.types import BudgetConfig
    turns=[DialogueTurn('old',(HumanMessage(content='a',id='o'),AIMessage(content='b',id='p')),('o','p')),DialogueTurn('new',(HumanMessage(content='a'*40,id='n'),AIMessage(content='b',id='q')),('n','q'))]
    selected=select_history(turns,HumanMessage(content='now',id='u'),counter=UnitCounter(),budget=BudgetConfig(40,8,2),system_messages=[])
    assert [m.id for m in selected.messages]==['u']

def test_exact_boundary_and_more_than_forty_messages_are_supported():
    from chatbot.emotion.budget import select_history
    from chatbot.emotion.history import DialogueTurn
    from chatbot.emotion.types import BudgetConfig
    turns=[DialogueTurn(str(i),(HumanMessage(content='a',id=f'u{i}'),AIMessage(content='b',id=f'a{i}')),(f'u{i}',f'a{i}')) for i in range(25)]
    current=HumanMessage(content='12345',id='now')
    result=select_history(turns,current,counter=UnitCounter(),budget=BudgetConfig(180,20,5),system_messages=[])
    assert len(result.messages)==51
    assert result.audit['chat_tokens']==106
    exact=select_history([],HumanMessage(content='12345',id='now'),counter=UnitCounter(),budget=BudgetConfig(10,2,1),system_messages=[])
    assert exact.audit['chat_tokens']==6
