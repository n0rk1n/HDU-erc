from fastapi.testclient import TestClient
from tests.emotion.helpers import emotion_runtime
from chatbot.db.connection import Database
from chatbot.emotion.types import ModelOutcome
from chatbot.llm.types import TokenUsage
from tests.emotion.helpers import create_app
from tests.api.helpers import OfflineModel
from tests.api.conftest import app_config


def test_app_accepts_explicit_offline_emotion_dependencies(app_config):
    runtime=emotion_runtime(Database(app_config.sqlite_db_path))
    with TestClient(create_app(config=app_config,model=OfflineModel(),emotion=runtime)) as client:
        response=client.post('/api/users/resolve',json={'identifier':'new-emotion'})
        assert response.status_code==200
        assert client.app.state.emotions is runtime.repository


def test_three_turns_persist_success_failure_and_complete_role_history(app_config):
    import json
    import sqlite3
    from uuid import uuid4
    from tests.api.helpers import parse_sse
    class Emotion:
        parameters={'model':'offline','temperature':0,'max_retries':0}
        def __init__(self):self.prompts=[]
        async def invoke(self,prompt):
            self.prompts.append(prompt)
            if len(self.prompts)==2:
                return ModelOutcome('partial invalid output',None,TokenUsage(9,None,None),None,{},
                    {'stage':'model','type':'TimeoutError','message':'deadline exceeded','http_status':504,'provider_request_id':'failed-r2'})
            return ModelOutcome(json.dumps({'primary_emotion':'neutral','confidence':.9,'secondary_emotions':[], 'evidence':'任务','reply_strategy':'直接回答','trajectory_note':'','safety_level':'normal'}),None,TokenUsage(10,4,14),'stop',{},None)
    model=Emotion();runtime=emotion_runtime(Database(app_config.sqlite_db_path),model)
    chat=OfflineModel()
    from tests.emotion_gate.helpers import gate_runtime
    from chatbot.emotion_gate.types import GatePolicy
    gate=gate_runtime(Database(app_config.sqlite_db_path),policy=GatePolicy(max_interval_turns=1))
    with TestClient(create_app(config=app_config,model=chat,emotion=runtime,gate=gate)) as client:
        user=client.post('/api/users/resolve',json={'identifier':'full-flow'}).json()['user']['id']
        requests=[str(uuid4()) for _ in range(3)]
        for i,request in enumerate(requests):
            response=client.post(f'/api/users/{user}/messages:stream',json={'request_id':request,'content':f'任务{i}'})
            assert parse_sse(response.text)[-1][0]=='done'
        response=client.post(f'/api/users/{user}/messages:stream',json={'request_id':requests[-1],'content':'任务2'})
        assert parse_sse(response.text)[-1][0]=='done'
    assert len(model.prompts)==3 and chat.calls==3
    assert [m.type for m in model.prompts[2]]==['system','human','ai','human','ai','human']
    assert [m.content for m in model.prompts[2][1:]]==['任务0','你好','任务1','你好','任务2']
    with sqlite3.connect(app_config.sqlite_db_path) as con:
        rows=con.execute('SELECT status,snapshot_json,error_json,raw_output,input_tokens,output_tokens FROM emotion_analyses ORDER BY created_at').fetchall()
        assert con.execute('SELECT count(*) FROM messages').fetchone()[0]==6
    assert [r[0] for r in rows]==['completed','failed','completed']
    error=json.loads(rows[1][2])
    assert error['provider_request_id']=='failed-r2' and error['http_status']==504
    assert rows[1][3]=='partial invalid output' and rows[1][4:]==(9,None)
    snapshot=json.loads(rows[2][1])
    assert [m['role'] for m in snapshot['role_history']]==['human','assistant','human','assistant','human']
    assert snapshot['model_parameters']['max_retries']==0
    assert len(snapshot['labels'])==28
    assert snapshot['budget']['chat_tokens']<=60000


def test_startup_recovers_pending_analysis_and_retains_snapshot(app_config):
    import sqlite3
    import json
    from uuid import uuid4
    runtime=emotion_runtime(Database(app_config.sqlite_db_path))
    with TestClient(create_app(config=app_config,model=OfflineModel(),emotion=runtime)) as client:
        identity=client.post('/api/users/resolve',json={'identifier':'restart-emotion'}).json()
        convo=identity['conversation']['id']
    request=str(uuid4());user_message=str(uuid4());assistant_message=str(uuid4());analysis=str(uuid4())
    with sqlite3.connect(app_config.sqlite_db_path) as con:
        for identifier,role,sequence,status in [(user_message,'user',1,'completed'),(assistant_message,'assistant',2,'pending')]:
            con.execute('INSERT INTO messages(id,conversation_id,request_id,sequence_no,role,status,content,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',(identifier,convo,request,sequence,role,status,'history','2026-01-01','2026-01-01'))
        con.execute('INSERT INTO emotion_analyses(id,conversation_id,request_id,user_message_id,status,snapshot_json,created_at) VALUES(?,?,?,?,?,?,?)',(analysis,convo,request,user_message,'running','{"model_parameters":{"model":"before-restart"}}','2026-01-01'))
    with TestClient(create_app(config=app_config,model=OfflineModel(),emotion=runtime)) as client:
        assert client.app.state.recovery_report.failed_analyses==1
    with sqlite3.connect(app_config.sqlite_db_path) as con:
        row=con.execute('SELECT status,snapshot_json,error_json FROM emotion_analyses WHERE id=?',(analysis,)).fetchone()
    assert row[0]=='failed'
    assert json.loads(row[1])['model_parameters']['model']=='before-restart'
    assert json.loads(row[2])['code']=='process_interrupted'
