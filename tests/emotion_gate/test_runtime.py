import asyncio
from dataclasses import replace
import pytest
from chatbot.db.messages import MessageRepository
from chatbot.services.identity import IdentityService
from tests.emotion_gate.helpers import gate_runtime,GateModel
from chatbot.emotion_gate.types import GatePolicy

async def run_new(database, runtime):
    from chatbot.emotion_gate.runtime import decide_emotion
    convo=(await IdentityService(database).resolve('runtime')).conversation
    turn=await MessageRepository(database).reserve_turn(convo.id,'r','hello')
    return await decide_emotion(runtime,conversation_id=convo.id,request_id='r',user_message_id=turn.user.id)

async def test_forced_skips_agent_and_replay(database):
    model=GateModel();runtime=gate_runtime(database,model)
    first=await run_new(database,runtime);again=await run_new(database,runtime)
    assert first.action=='analyze' and first.reason=='first_turn' and first.id==again.id
    assert model.prompts==[]

@pytest.mark.parametrize('action',['analyze','skip'])
async def test_failure_policy_and_audited_retries(database,action):
    model=GateModel(['bad',TimeoutError('deadline')])
    runtime=gate_runtime(database,model,GatePolicy(force_first_analysis=False,gate_failure_action=action,model_retry_count=1))
    row=await run_new(database,runtime)
    assert row.action==action and row.reason=='gate_failed'
    attempts=await runtime.repository.list_attempts(row.id)
    assert len(attempts)==2 and all(a['status']=='failed' for a in attempts)
    assert attempts[0]['facts']['raw_output']=='bad'
    assert attempts[1]['facts']['error']['type']=='TimeoutError'
    assert attempts[0]['snapshot']['prompt'][-1]['content']=='hello'

async def test_stable_and_cancel(database):
    runtime=gate_runtime(database,GateModel(),GatePolicy(force_first_analysis=False))
    row=await run_new(database,runtime)
    assert row.action=='skip' and row.reason=='unchanged'
    assert (await runtime.repository.list_attempts(row.id))[0]['facts']['result']['should_analyze'] is False

async def test_cancellation_is_durable(database):
    runtime=gate_runtime(database,GateModel([asyncio.CancelledError()]),GatePolicy(force_first_analysis=False))
    with pytest.raises(asyncio.CancelledError):await run_new(database,runtime)
    async with database.connect() as con:
        row=await (await con.execute('SELECT id,status FROM emotion_gate_decisions')).fetchone()
    assert row[1]=='failed'
    assert (await runtime.repository.list_attempts(row[0]))[0]['status']=='failed'

async def test_storage_failure_does_not_retry_or_fallback(database):
    from chatbot.db.emotion_gates import GateRepository
    class Broken(GateRepository):
        async def finish_attempt(self,*args,**kwargs):raise RuntimeError('disk')
    runtime=replace(gate_runtime(database,GateModel(),GatePolicy(force_first_analysis=False)),repository=Broken(database))
    with pytest.raises(RuntimeError,match='disk'):await run_new(database,runtime)
    assert len(runtime.model.prompts)==1
