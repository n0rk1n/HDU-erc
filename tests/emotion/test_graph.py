import json
import pytest
from langchain_core.messages import HumanMessage
from chatbot.db.messages import MessageRepository
from chatbot.db.emotions import EmotionRepository
from chatbot.services.identity import IdentityService
from chatbot.emotion.config import load_taxonomy, CONFIG_ROOT
from chatbot.emotion.prompt import load_examples
from chatbot.emotion.types import BudgetConfig,ModelOutcome
from chatbot.llm.types import TokenUsage

class Counter:
    identity='fake';version='1'
    def count(self,messages): return sum(len(m.content)+1 for m in messages)

@pytest.mark.parametrize('raw,error,status', [
    ('{"primary_emotion":"neutral","confidence":0.9,"secondary_emotions":[],"evidence":"你好","reply_strategy":"问候","trajectory_note":"","safety_level":"normal"}',None,'completed'),
    ('not json',None,'failed'),
    (None,{'stage':'model','type':'TimeoutError','message':'timeout'},'failed')])
async def test_graph_persists_before_call_and_failure_keeps_facts(database,raw,error,status):
    from chatbot.emotion.graph import EmotionRuntime,build_emotion_graph
    messages=MessageRepository(database);repo=EmotionRepository(database)
    convo=(await IdentityService(database).resolve('subgraph')).conversation
    turn=await messages.reserve_turn(convo.id,'request','你好')
    class Model:
        calls=0
        parameters={'model':'fake','max_retries':0}
        async def invoke(self,prompt):
            row=await repo.find_for_message(turn.user.id)
            assert row.status=='running'
            assert row.snapshot_json['model_parameters']['max_retries']==0
            assert row.snapshot_json['prompt'][-1]['content']=='你好'
            self.calls+=1
            return ModelOutcome(raw,None,TokenUsage(10,2,12),'stop',{},error)
    model=Model();taxonomy=load_taxonomy()
    runtime=EmotionRuntime(repo,messages,model,Counter(),BudgetConfig(100000,1000,256),taxonomy,load_examples(CONFIG_ROOT/'emotion_examples.json',taxonomy))
    graph=build_emotion_graph(runtime)
    state={'conversation_id':convo.id,'request_id':'request','user_message_id':turn.user.id}
    result=await graph.ainvoke(state)
    row=await repo.get(result['analysis_id'])
    assert row.status==status and row.raw_output==raw
    assert row.total_tokens==12
    if status=='failed': assert row.error_json
    await graph.ainvoke(state)
    assert model.calls==1

async def test_oversize_current_input_is_audited_without_model_call(database):
    from chatbot.emotion.graph import EmotionRuntime,build_emotion_graph
    from tests.emotion.helpers import emotion_runtime
    from dataclasses import replace
    messages=MessageRepository(database)
    convo=(await IdentityService(database).resolve('too-long')).conversation
    turn=await messages.reserve_turn(convo.id,'long','x'*100)
    class Never:
        parameters={'model':'never'}
        async def invoke(self,prompt):raise AssertionError('must not call')
    runtime=replace(emotion_runtime(database,Never()),budget=BudgetConfig(100,10,5))
    result=await build_emotion_graph(runtime).ainvoke({'conversation_id':convo.id,'request_id':'long','user_message_id':turn.user.id})
    row=await runtime.repository.get(result['analysis_id'])
    assert row.status=='failed'
    assert row.error_json['code']=='context_budget_exceeded'
    assert row.snapshot_json['budget']['history_token_cap']==60
    assert row.input_tokens is None
