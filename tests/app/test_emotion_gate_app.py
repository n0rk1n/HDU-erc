import asyncio
import sqlite3
import json
from fastapi.testclient import TestClient
from chatbot.web import create_app
from chatbot.db.connection import Database
from chatbot.db.schema import initialize_schema
from chatbot.db.messages import MessageRepository
from chatbot.services.identity import IdentityService
from tests.api.conftest import app_config
from tests.api.helpers import OfflineModel,parse_sse
from tests.emotion.helpers import emotion_runtime
from tests.emotion_gate.helpers import gate_runtime
from tests.graph.test_emotion_gate import SuccessEmotion


def test_restart_keeps_baseline_and_skips_stable_turn(app_config):
    db=Database(app_config.sqlite_db_path);model=SuccessEmotion()
    emotion=emotion_runtime(db,model);gate=gate_runtime(db)
    for number in (1,2):
        with TestClient(create_app(config=app_config,model=OfflineModel(),emotion=emotion,gate=gate)) as client:
            user=client.post('/api/users/resolve',json={'identifier':'restart-gate'}).json()['user']['id']
            from uuid import uuid4
            result=client.post(f'/api/users/{user}/messages:stream',json={'request_id':str(uuid4()),'content':'hi'})
            assert parse_sse(result.text)[-1][0]=='done'
    assert len(model.prompts)==1
    with sqlite3.connect(db.path) as con:
        assert con.execute('SELECT count(*) FROM emotion_gate_decisions').fetchone()[0]==2
        assert con.execute('SELECT count(*) FROM emotion_analyses').fetchone()[0]==1


def test_startup_recovers_pending_gate_without_external_calls(app_config):
    db=Database(app_config.sqlite_db_path);gate=gate_runtime(db)
    async def prepare():
        await initialize_schema(db)
        convo=(await IdentityService(db).resolve('interrupted')).conversation
        turn=await MessageRepository(db).reserve_turn(convo.id,'r','hello')
        row,_=await gate.repository.reserve(convo.id,'r',turn.user.id)
        await gate.repository.start(row.id,snapshot={'saved':True})
        await gate.repository.start_attempt(row.id,snapshot={'input':'saved'})
        return row.id
    rowid=asyncio.run(prepare())
    with TestClient(create_app(config=app_config,model=OfflineModel(),emotion=emotion_runtime(db),gate=gate)) as client:
        assert client.app.state.recovery_report.failed_gate_decisions==1
    row=asyncio.run(gate.repository.get(rowid))
    assert row.status=='failed' and row.snapshot['saved']
    assert asyncio.run(gate.repository.list_attempts(rowid))[0]['status']=='failed'
    assert gate.model.prompts==[]
