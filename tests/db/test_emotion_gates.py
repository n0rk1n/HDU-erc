import pytest
from chatbot.db.messages import MessageRepository
from chatbot.db.emotions import EmotionRepository
from chatbot.services.identity import IdentityService
from chatbot.core.errors import InvalidMessageState

async def test_gate_facts_and_replay(database):
    from chatbot.db.emotion_gates import GateRepository
    conversation=(await IdentityService(database).resolve('gate')).conversation
    messages=MessageRepository(database);repo=GateRepository(database)
    first=await messages.reserve_turn(conversation.id,'1','hi')
    row,created=await repo.reserve(conversation.id,'1',first.user.id)
    repeated,isnew=await repo.reserve(conversation.id,'1',first.user.id)
    assert created and not isnew and repeated.id==row.id
    with pytest.raises(InvalidMessageState):await repo.reserve(conversation.id,'1',first.assistant.id)
    analysis,_=await EmotionRepository(database).reserve(conversation.id,'1',first.user.id)
    await EmotionRepository(database).finish(analysis.id,status='completed',facts={'result':{'primary_emotion':'sad'}})
    second=await messages.reserve_turn(conversation.id,'2','again')
    facts=await repo.facts(conversation.id,second.user.id)
    assert facts.current_turn==2 and facts.last_success_turn==1 and not facts.latest_analysis_failed
    await messages.reserve_turn(conversation.id,'future','future')
    assert (await repo.facts(conversation.id,second.user.id)).current_turn==2
    await repo.start(row.id,snapshot={'policy':{'max_interval_turns':15}})
    attempt=await repo.start_attempt(row.id,snapshot={'prompt':'hi'})
    await repo.finish_attempt(attempt,status='failed',facts={'error':{'code':'timeout'}})
    await repo.finish(row.id,action='analyze',reason='gate_failed',error={'code':'timeout'})
    await repo.attach_analysis(row.id,analysis.id)
    final=await GateRepository(database).get(row.id)
    assert final.action=='analyze' and final.analysis_id==analysis.id
    assert (await repo.list_attempts(row.id))[0]['facts']['error']['code']=='timeout'
    with pytest.raises(InvalidMessageState):await repo.start_attempt(row.id,snapshot={})

async def test_v2_upgrade_preserves_analysis_and_recovers_attempt(tmp_path):
    from chatbot.db.schema import V1_DDL_STATEMENTS,V2_DDL_STATEMENTS,initialize_schema
    from chatbot.db.connection import Database
    from chatbot.db.emotion_gates import GateRepository
    db=Database(tmp_path/'v2.db')
    async with db.transaction() as con:
        for ddl in (*V1_DDL_STATEMENTS,*V2_DDL_STATEMENTS):await con.execute(ddl)
        await con.execute('PRAGMA user_version=2')
    convo=(await IdentityService(db).resolve('old')).conversation
    # Seed an actual v2 row, without asking the current repository to read an old schema.
    async with db.transaction() as con:
        await con.execute("""INSERT INTO messages(id,conversation_id,request_id,sequence_no,role,status,content,created_at,updated_at)
            VALUES ('old-user',?,'r',1,'user','completed','hello','2026-09-03','2026-09-03')""",(convo.id,))
    analysis,_=await EmotionRepository(db).reserve(convo.id,'r','old-user')
    await initialize_schema(db);await initialize_schema(db)
    assert (await EmotionRepository(db).get(analysis.id)).status=='pending'
    repo=GateRepository(db);row,_=await repo.reserve(convo.id,'r','old-user')
    await repo.start(row.id,snapshot={'retained':'yes'})
    await repo.start_attempt(row.id,snapshot={'input':'saved'})
    assert await repo.fail_interrupted(cutoff='9999')==1
    assert await repo.fail_interrupted(cutoff='9999')==0
    assert (await repo.get(row.id)).error['code']=='process_interrupted'
    assert (await repo.list_attempts(row.id))[0]['status']=='failed'
