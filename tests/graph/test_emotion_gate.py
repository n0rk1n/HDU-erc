import json
from dataclasses import replace
import pytest
from chatbot.graph import compile_turn_graph,NodeDependencies
from chatbot.llm.types import ModelDelta,TokenUsage
from chatbot.emotion.types import ModelOutcome
from tests.emotion.helpers import emotion_runtime
from tests.emotion_gate.helpers import gate_runtime,GateModel
from chatbot.emotion_gate.types import GatePolicy
from tests.graph.test_turn_graph import messages,saver,_reserved_context,_config,DeterministicModel

class SuccessEmotion:
    parameters={'model':'offline','max_retries':0}
    def __init__(self): self.prompts=[]
    async def invoke(self,prompt):
        self.prompts.append(prompt)
        return ModelOutcome(json.dumps({'primary_emotion':'sad','confidence':0.9,'secondary_emotions':[],
            'evidence':'sad','reply_strategy':'listen','trajectory_note':'','safety_level':'normal'}),None,TokenUsage(),'stop',{},None)

@pytest.mark.parametrize('interval,total,want_calls',[(15,16,2),(2,5,3),(1,3,3)])
async def test_interval_routing_replay_and_historical_context(database,messages,saver,interval,total,want_calls):
    model=SuccessEmotion();gate_model=GateModel();emotion=emotion_runtime(database,model)
    gate=gate_runtime(database,gate_model,GatePolicy(max_interval_turns=interval))
    chat=DeterministicModel([ModelDelta(content='reply')])
    graph=compile_turn_graph(NodeDependencies(messages=messages,model=chat,emotion=emotion,gate=gate),saver)
    for i in range(total):
        context,state=await _reserved_context(database,messages,'sequence')
        result=await graph.ainvoke(state,_config(context),context=context)
        assert result['phase']=='completed'
        await graph.ainvoke(state,_config(context),context=context)
    assert len(model.prompts)==want_calls and chat.calls==total
    assert len(gate_model.prompts)==total-want_calls
    if interval>1:
        assert '"is_historical": true' in chat.prompts[1][0].content
        assert '"elapsed_turns": 1' in chat.prompts[1][0].content
    assert (await gate.repository.facts(context.conversation_id,context.user_message_id)).current_turn==total

async def test_early_change_resets_interval(database,messages,saver):
    emotion_model=SuccessEmotion()
    gate=gate_runtime(database,GateModel(['{"should_analyze":true,"reason":"changed"}','{"should_analyze":false,"reason":"stable"}']),GatePolicy(max_interval_turns=2))
    graph=compile_turn_graph(NodeDependencies(messages=messages,model=DeterministicModel([ModelDelta(content='r')]),emotion=emotion_runtime(database,emotion_model),gate=gate),saver)
    for i in range(4):
        context,state=await _reserved_context(database,messages,'early')
        assert (await graph.ainvoke(state,_config(context),context=context))['phase']=='completed'
    assert len(emotion_model.prompts)==3
    async with database.connect() as con:
        reasons=await (await con.execute('SELECT reason FROM emotion_gate_decisions ORDER BY created_at')).fetchall()
    assert [r[0] for r in reasons]==['first_turn','emotion_changed','unchanged','interval_reached']

@pytest.mark.parametrize('reuse_failure',[False,True])
async def test_failed_analysis_does_not_reset_and_retries_next_turn(database,messages,saver,reuse_failure):
    class FailSecond(SuccessEmotion):
        async def invoke(self,prompt):
            outcome=await super().invoke(prompt)
            if len(self.prompts)==2:
                return replace(outcome,error={'stage':'model','message':'timeout'})
            return outcome
    model=FailSecond();gate_model=GateModel(['{"should_analyze":true,"reason":"change"}'])
    gate=gate_runtime(database,gate_model,GatePolicy(reuse_last_success_on_analysis_failure=reuse_failure))
    chat=DeterministicModel([ModelDelta(content='reply')])
    graph=compile_turn_graph(NodeDependencies(messages=messages,model=chat,emotion=emotion_runtime(database,model),gate=gate),saver)
    for i in range(3):
        context,state=await _reserved_context(database,messages,'fail-retry')
        assert (await graph.ainvoke(state,_config(context),context=context))['phase']=='completed'
    assert len(model.prompts)==3 and len(gate_model.prompts)==1
    assert ('Emotion Context' in chat.prompts[1][0].content)==reuse_failure
    if reuse_failure: assert '"is_historical": true' in chat.prompts[1][0].content
    async with database.connect() as con:
        rows=await (await con.execute('SELECT reason,snapshot_json FROM emotion_gate_decisions ORDER BY created_at')).fetchall()
    assert rows[2][0]=='previous_analysis_failed'
    assert json.loads(rows[2][1])['facts']['last_success_turn']==1

async def test_skip_without_reuse_and_foreign_reference_rejected(database,messages,saver):
    from chatbot.emotion_gate.context import build_reply_emotion_context
    from chatbot.core.errors import InvalidMessageState
    gate=gate_runtime(database,policy=GatePolicy(reuse_last_success_on_skip=False))
    chat=DeterministicModel([ModelDelta(content='reply')])
    emotion=emotion_runtime(database,SuccessEmotion())
    graph=compile_turn_graph(NodeDependencies(messages=messages,model=chat,emotion=emotion,gate=gate),saver)
    contexts=[];results=[]
    for name in ('own','own','foreign'):
        context,state=await _reserved_context(database,messages,name)
        contexts.append(context);results.append(await graph.ainvoke(state,_config(context),context=context))
    assert 'Emotion Context' not in chat.prompts[1][0].content
    decision=await gate.repository.get(results[1]['gate_decision_id'])
    foreign=await emotion.repository.get(results[2]['emotion_analysis_id'])
    snapshot=decision.snapshot
    snapshot['policy']['reuse_last_success_on_skip']=True
    snapshot['baseline']['analysis_id']=foreign.id
    snapshot['baseline']['user_message_id']=foreign.user_message_id
    async with database.transaction() as con:
        await con.execute('UPDATE emotion_gate_decisions SET snapshot_json=? WHERE id=?',(json.dumps(snapshot),decision.id))
    with pytest.raises(InvalidMessageState,match='historical emotion identity'):
        await build_reply_emotion_context(gate=gate,decision_id=decision.id,analysis_id=None,
            conversation_id=contexts[1].conversation_id,user_message_id=contexts[1].user_message_id)

async def test_gate_storage_failure_stops_reply(database,messages,saver):
    from chatbot.db.emotion_gates import GateRepository
    class Broken(GateRepository):
        async def start(self,*args,**kwargs):raise RuntimeError('storage unavailable')
    gate=replace(gate_runtime(database),repository=Broken(database))
    chat=DeterministicModel([ModelDelta(content='must not run')]);emotion_model=SuccessEmotion()
    graph=compile_turn_graph(NodeDependencies(messages=messages,model=chat,emotion=emotion_runtime(database,emotion_model),gate=gate),saver)
    context,state=await _reserved_context(database,messages,'bad-store')
    result=await graph.ainvoke(state,_config(context),context=context)
    assert result['phase']=='failed' and result['error_code']=='database_error'
    assert chat.calls==0 and emotion_model.prompts==[]
