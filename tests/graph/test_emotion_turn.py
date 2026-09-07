import json
import pytest
from chatbot.graph import compile_turn_graph
from tests.emotion.helpers import dependencies as NodeDependencies
from chatbot.llm.types import ModelDelta,TokenUsage
from chatbot.emotion.types import ModelOutcome
from tests.emotion.helpers import emotion_runtime
from tests.graph.test_turn_graph import messages,saver,_reserved_context,_config,DeterministicModel

@pytest.mark.parametrize('fail',[False,True])
async def test_emotion_is_durable_before_reply_and_failure_continues(database,messages,saver,fail):
    context,state=await _reserved_context(database,messages,'emotion-turn')
    class Emotion:
        parameters={'model':'fake','max_retries':0}
        calls=0
        async def invoke(self,prompt):
            self.calls+=1
            raw=json.dumps({'primary_emotion':'sad','confidence':.8,'secondary_emotions':[], 'evidence':'问题','reply_strategy':'倾听','trajectory_note':'','safety_level':'normal'})
            return ModelOutcome(raw,None,TokenUsage(), 'stop',{}, {'message':'timeout','stage':'model'} if fail else None)
    model=Emotion();emotion=emotion_runtime(database,model)
    class Chat(DeterministicModel):
        async def stream(self,prompt):
            row=await emotion.repository.find_for_message(context.user_message_id)
            assert row.status==('failed' if fail else 'completed')
            assert ('Emotion Context' in prompt[0].content)==(not fail)
            async for delta in super().stream(prompt):yield delta
    chat=Chat([ModelDelta(content='回复')])
    graph=compile_turn_graph(NodeDependencies(messages=messages,model=chat,emotion=emotion),saver)
    result=await graph.ainvoke(state,_config(context),context=context)
    assert result['phase']=='completed'
    await graph.ainvoke(state,_config(context),context=context)
    assert model.calls==1

async def test_analysis_persistence_failure_stops_reply_and_cleans_run(database,messages,saver):
    from dataclasses import replace
    from chatbot.db.emotions import EmotionRepository
    from chatbot.graph.nodes import TurnNodes
    from langgraph.runtime import Runtime
    context,state=await _reserved_context(database,messages,'db-fail')
    class BrokenRepository(EmotionRepository):
        async def start(self,*args,**kwargs):
            raise RuntimeError('storage unavailable')
    emotion=replace(emotion_runtime(database),repository=BrokenRepository(database))
    chat=DeterministicModel([ModelDelta(content='must not run')])
    nodes=TurnNodes(NodeDependencies(messages=messages,model=chat,emotion=emotion))
    runtime=Runtime(context=context)
    prepared={**state,**await nodes.prepare_turn(state,runtime)}
    result=await nodes.analyze_emotion(prepared,runtime)
    assert result['phase']=='failed'
    assert result['error_code']=='database_error'
    assert nodes._runs=={}
    assistant=await messages.find_assistant(context.conversation_id,context.request_id)
    assert assistant.status=='failed' and assistant.content==''
    assert json.loads(assistant.trace_json)['model_calls']==[]

async def test_emotion_cancellation_propagates_even_if_audit_write_fails(database,messages,saver):
    import asyncio
    from dataclasses import replace
    from chatbot.db.emotions import EmotionRepository
    from chatbot.graph.nodes import TurnNodes
    from langgraph.runtime import Runtime
    context,state=await _reserved_context(database,messages,'cancel-emotion')
    class Cancel:
        parameters={'model':'cancel'}
        async def invoke(self,prompt):raise asyncio.CancelledError()
    class FailingFinish(EmotionRepository):
        async def finish(self,*args,**kwargs):raise RuntimeError('disk unavailable')
    emotion=replace(emotion_runtime(database,Cancel()),repository=FailingFinish(database))
    nodes=TurnNodes(NodeDependencies(messages=messages,model=DeterministicModel([]),emotion=emotion))
    runtime=Runtime(context=context)
    prepared={**state,**await nodes.prepare_turn(state,runtime)}
    with pytest.raises(asyncio.CancelledError):
        await nodes.analyze_emotion(prepared,runtime)
    assert nodes._runs=={}
