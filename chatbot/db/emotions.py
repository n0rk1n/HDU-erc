"""Business audit records, independent of graph checkpoints."""
from __future__ import annotations
import json
from dataclasses import dataclass
from uuid import uuid4
import aiosqlite
from chatbot.core.errors import InvalidMessageState
from chatbot.core.time import utc_now
from chatbot.db.connection import Database

@dataclass(frozen=True)
class EmotionAnalysis:
    id: str
    conversation_id: str
    request_id: str
    user_message_id: str
    status: str
    snapshot_json: dict
    result_json: dict | None
    raw_output: str | None
    reasoning_content: str | None
    response_metadata_json: dict | None
    error_json: dict | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    finish_reason: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None

def _decode(row):
    if row is None:
        return None
    data = dict(row)
    for key in ('snapshot_json','result_json','response_metadata_json','error_json'):
        data[key] = json.loads(data[key]) if data[key] is not None else None
    return EmotionAnalysis(**data)

def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False) if value is not None else None

class EmotionRepository:
    def __init__(self, database: Database):
        self.database = database

    async def reserve(self, conversation_id, request_id, user_message_id):
        async with self.database.transaction(immediate=True) as con:
            con.row_factory = aiosqlite.Row
            user = await (await con.execute('SELECT * FROM messages WHERE id=?', (user_message_id,))).fetchone()
            if (user is None or user['conversation_id'] != conversation_id or user['request_id'] != request_id
                or user['role'] != 'user' or user['status'] != 'completed'):
                raise InvalidMessageState('invalid emotion analysis binding')
            row = await (await con.execute('SELECT * FROM emotion_analyses WHERE user_message_id=?', (user_message_id,))).fetchone()
            if row:
                return _decode(row), False
            identifier = str(uuid4())
            await con.execute('INSERT INTO emotion_analyses(id,conversation_id,request_id,user_message_id,status,created_at) VALUES(?,?,?,?,?,?)',
                              (identifier,conversation_id,request_id,user_message_id,'pending',utc_now()))
            row = await (await con.execute('SELECT * FROM emotion_analyses WHERE id=?', (identifier,))).fetchone()
            return _decode(row), True

    async def get(self, analysis_id):
        async with self.database.connect() as con:
            con.row_factory = aiosqlite.Row
            return _decode(await (await con.execute('SELECT * FROM emotion_analyses WHERE id=?', (analysis_id,))).fetchone())

    async def find_for_message(self, user_message_id):
        async with self.database.connect() as con:
            con.row_factory = aiosqlite.Row
            return _decode(await (await con.execute('SELECT * FROM emotion_analyses WHERE user_message_id=?', (user_message_id,))).fetchone())

    async def start(self, analysis_id, *, snapshot):
        async with self.database.transaction(immediate=True) as con:
            cursor = await con.execute("UPDATE emotion_analyses SET status='running', snapshot_json=?,started_at=? WHERE id=? AND status='pending'", (_json(snapshot),utc_now(),analysis_id))
            if cursor.rowcount != 1:
                raise InvalidMessageState('analysis is not pending')

    async def finish(self, analysis_id, *, status, facts):
        if status not in ('completed','failed'):
            raise ValueError('analysis requires terminal status')
        async with self.database.transaction(immediate=True) as con:
            row = await (await con.execute('SELECT status FROM emotion_analyses WHERE id=?', (analysis_id,))).fetchone()
            if row is None:
                raise InvalidMessageState('analysis does not exist')
            if row[0] in ('completed','failed'):
                return
            mapping = {'result':'result_json','metadata':'response_metadata_json','error':'error_json','snapshot':'snapshot_json'}
            updates = {'status':status, 'completed_at':utc_now()}
            for key,column in mapping.items():
                if key in facts:
                    updates[column] = _json(facts[key])
            for key in ('raw_output','reasoning_content','input_tokens','output_tokens','total_tokens','latency_ms','finish_reason'):
                if key in facts:
                    updates[key] = facts[key]
            await con.execute('UPDATE emotion_analyses SET '+','.join(f'{k}=?' for k in updates)+' WHERE id=?', (*updates.values(),analysis_id))

    async def fail_interrupted(self, *, cutoff):
        async with self.database.transaction(immediate=True) as con:
            rows = await (await con.execute("SELECT id,error_json FROM emotion_analyses WHERE status IN ('pending','running') AND created_at<=?", (cutoff,))).fetchall()
            for identifier,previous in rows:
                error = {'stage':'recovery','code':'process_interrupted','message':'emotion analysis interrupted'}
                if previous:
                    error['previous'] = json.loads(previous)
                await con.execute("UPDATE emotion_analyses SET status='failed',error_json=?,completed_at=? WHERE id=?", (_json(error),utc_now(),identifier))
            return len(rows)

    async def recent_labels(self, conversation_id, *, before_sequence, limit=3):
        async with self.database.connect() as con:
            rows = await (await con.execute("SELECT a.result_json FROM emotion_analyses a JOIN messages m ON m.id=a.user_message_id WHERE a.conversation_id=? AND m.sequence_no<? AND a.status='completed' ORDER BY m.sequence_no DESC", (conversation_id,before_sequence))).fetchall()
        labels=[]
        for row in rows:
            label=json.loads(row[0])['primary_emotion']
            if label not in labels:
                labels.append(label)
            if len(labels)>=limit:
                break
        return labels
