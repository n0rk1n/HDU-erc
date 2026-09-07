"""Durable scheduling decisions and separately audited model attempts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4
import aiosqlite
from chatbot.core.errors import InvalidMessageState
from chatbot.core.time import utc_now
from chatbot.db.connection import Database


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class GateFacts:
    current_turn: int
    last_success_turn: int | None
    last_success_analysis_id: str | None
    latest_analysis_failed: bool


@dataclass(frozen=True)
class GateDecision:
    id: str
    conversation_id: str
    request_id: str
    user_message_id: str
    status: str
    action: str | None
    reason: str | None
    snapshot: dict
    error: dict | None
    analysis_id: str | None
    created_at: str
    completed_at: str | None


def _decode(row):
    if row is None:
        return None
    data = dict(row)
    data['snapshot'] = json.loads(data.pop('snapshot_json'))
    error = data.pop('error_json')
    data['error'] = json.loads(error) if error else None
    return GateDecision(**data)


class GateRepository:
    def __init__(self, database: Database):
        self.database = database

    async def _user(self, con, conversation_id, user_message_id):
        row = await (await con.execute(
            "SELECT * FROM messages WHERE id=? AND conversation_id=? AND role='user' AND status='completed'",
            (user_message_id, conversation_id))).fetchone()
        if row is None:
            raise InvalidMessageState('invalid gate user binding')
        return row

    async def reserve(self, conversation_id, request_id, user_message_id):
        async with self.database.transaction(immediate=True) as con:
            con.row_factory = aiosqlite.Row
            user = await self._user(con, conversation_id, user_message_id)
            if user['request_id'] != request_id:
                raise InvalidMessageState('invalid gate request binding')
            row = await (await con.execute('SELECT * FROM emotion_gate_decisions WHERE user_message_id=?', (user_message_id,))).fetchone()
            if row:
                return _decode(row), False
            identifier = str(uuid4())
            await con.execute('INSERT INTO emotion_gate_decisions(id,conversation_id,request_id,user_message_id,status,created_at) VALUES(?,?,?,?,?,?)',
                              (identifier, conversation_id, request_id, user_message_id, 'pending', utc_now()))
            return _decode(await (await con.execute('SELECT * FROM emotion_gate_decisions WHERE id=?', (identifier,))).fetchone()), True

    async def get(self, identifier):
        async with self.database.connect() as con:
            con.row_factory = aiosqlite.Row
            return _decode(await (await con.execute('SELECT * FROM emotion_gate_decisions WHERE id=?', (identifier,))).fetchone())

    async def facts(self, conversation_id, user_message_id):
        async with self.database.connect() as con:
            con.row_factory = aiosqlite.Row
            user = await self._user(con, conversation_id, user_message_id)
            async def ordinal(sequence):
                return (await (await con.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=? AND role='user' AND sequence_no<=?", (conversation_id, sequence))).fetchone())[0]
            current = await ordinal(user['sequence_no'])
            analyses = await (await con.execute("SELECT a.id,a.status,m.sequence_no FROM emotion_analyses a JOIN messages m ON m.id=a.user_message_id WHERE a.conversation_id=? AND m.sequence_no<? ORDER BY m.sequence_no DESC", (conversation_id, user['sequence_no']))).fetchall()
            success = next((a for a in analyses if a['status'] == 'completed'), None)
            return GateFacts(current, await ordinal(success['sequence_no']) if success else None,
                             success['id'] if success else None, bool(analyses and analyses[0]['status'] == 'failed'))

    async def start(self, identifier, *, snapshot):
        async with self.database.transaction(immediate=True) as con:
            cursor = await con.execute("UPDATE emotion_gate_decisions SET status='running',snapshot_json=? WHERE id=? AND status='pending'", (_json(snapshot), identifier))
            if cursor.rowcount != 1:
                raise InvalidMessageState('gate is not pending')

    async def finish(self, identifier, *, action, reason, error=None):
        if action not in ('analyze', 'skip'):
            raise ValueError('invalid gate action')
        async with self.database.transaction(immediate=True) as con:
            cursor = await con.execute("UPDATE emotion_gate_decisions SET status='completed',action=?,reason=?,error_json=?,completed_at=? WHERE id=? AND status='running'", (action, reason, _json(error) if error else None, utc_now(), identifier))
            if cursor.rowcount != 1:
                raise InvalidMessageState('gate is not running')

    async def attach_analysis(self, identifier, analysis_id):
        async with self.database.transaction(immediate=True) as con:
            cursor = await con.execute("""UPDATE emotion_gate_decisions SET analysis_id=? WHERE id=? AND action='analyze'
                AND status='completed' AND (analysis_id IS NULL OR analysis_id=?)
                AND EXISTS(SELECT 1 FROM emotion_analyses a WHERE a.id=?
                AND a.user_message_id=emotion_gate_decisions.user_message_id
                AND a.conversation_id=emotion_gate_decisions.conversation_id)""", (analysis_id, identifier, analysis_id, analysis_id))
            if cursor.rowcount != 1:
                raise InvalidMessageState('invalid gate analysis binding')

    async def start_attempt(self, identifier, *, snapshot):
        async with self.database.transaction(immediate=True) as con:
            row = await (await con.execute('SELECT status FROM emotion_gate_decisions WHERE id=?', (identifier,))).fetchone()
            if row is None or row[0] != 'running':
                raise InvalidMessageState('gate is not running')
            number = (await (await con.execute('SELECT COALESCE(MAX(attempt_no),0)+1 FROM emotion_gate_attempts WHERE decision_id=?', (identifier,))).fetchone())[0]
            attempt = str(uuid4())
            await con.execute("INSERT INTO emotion_gate_attempts(id,decision_id,attempt_no,status,snapshot_json,created_at) VALUES(?,?,?,'running',?,?)", (attempt, identifier, number, _json(snapshot), utc_now()))
            return attempt

    async def finish_attempt(self, identifier, *, status, facts):
        if status not in ('completed', 'failed'):
            raise ValueError('attempt requires terminal status')
        async with self.database.transaction(immediate=True) as con:
            cursor = await con.execute("UPDATE emotion_gate_attempts SET status=?,facts_json=?,completed_at=? WHERE id=? AND status='running'", (status, _json(facts), utc_now(), identifier))
            if cursor.rowcount != 1:
                raise InvalidMessageState('attempt is not running')

    async def list_attempts(self, identifier):
        async with self.database.connect() as con:
            con.row_factory = aiosqlite.Row
            rows = await (await con.execute('SELECT * FROM emotion_gate_attempts WHERE decision_id=? ORDER BY attempt_no', (identifier,))).fetchall()
            return [{**dict(row), 'snapshot': json.loads(row['snapshot_json']), 'facts': json.loads(row['facts_json'])} for row in rows]

    async def interrupt(self, identifier):
        async with self.database.transaction(immediate=True) as con:
            await self._interrupt(con, identifier)

    async def _interrupt(self, con, identifier):
        error = {'stage': 'recovery', 'code': 'process_interrupted', 'message': 'emotion gate interrupted'}
        await con.execute("UPDATE emotion_gate_attempts SET status='failed',facts_json=?,completed_at=? WHERE decision_id=? AND status='running'", (_json({'error': error}), utc_now(), identifier))
        await con.execute("UPDATE emotion_gate_decisions SET status='failed',error_json=?,completed_at=? WHERE id=? AND status IN ('pending','running')", (_json(error), utc_now(), identifier))

    async def fail_interrupted(self, *, cutoff):
        async with self.database.transaction(immediate=True) as con:
            rows = await (await con.execute("SELECT id FROM emotion_gate_decisions WHERE status IN ('pending','running') AND created_at<=?", (cutoff,))).fetchall()
            for row in rows:
                await self._interrupt(con, row[0])
            return len(rows)
