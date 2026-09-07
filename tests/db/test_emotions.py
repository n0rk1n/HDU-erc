import pytest
from chatbot.db.messages import MessageRepository
from chatbot.services.identity import IdentityService
from chatbot.core.errors import InvalidMessageState

async def test_analysis_audit_is_bound_idempotent_and_durable(database):
    from chatbot.db.emotions import EmotionRepository
    conversation = (await IdentityService(database).resolve('audit')).conversation
    turn = await MessageRepository(database).reserve_turn(conversation.id, 'r1', '你好')
    repo = EmotionRepository(database)
    row, created = await repo.reserve(conversation.id, 'r1', turn.user.id)
    again, repeated = await repo.reserve(conversation.id, 'r1', turn.user.id)
    assert created and not repeated and row.id == again.id
    with pytest.raises(InvalidMessageState):
        await repo.reserve(conversation.id, 'r1', turn.assistant.id)
    await repo.start(row.id, snapshot={'prompt': [{'role':'user','content':'你好'}]})
    assert (await repo.get(row.id)).status == 'running'
    await repo.finish(row.id, status='failed', facts={'error':{'stage':'model','message':'timeout'},'raw_output':'partial'})
    await repo.finish(row.id, status='completed', facts={'result':{'primary_emotion':'sad'}})
    final = await repo.get(row.id)
    assert final.status == 'failed'
    assert final.raw_output == 'partial'
    assert final.error_json['message'] == 'timeout'
    assert final.input_tokens is None
    assert final.snapshot_json['prompt'][0]['content'] == '你好'

async def test_v1_migration_preserves_rows(tmp_path):
    from chatbot.db.schema import V1_DDL_STATEMENTS, initialize_schema
    from chatbot.db.connection import Database
    db = Database(tmp_path/'old.sqlite3')
    async with db.transaction() as con:
        for ddl in V1_DDL_STATEMENTS:
            await con.execute(ddl)
        await con.execute("INSERT INTO users(identifier,created_at,updated_at) VALUES('old','now','now')")
        await con.execute('PRAGMA user_version=1')
    await initialize_schema(db)
    await initialize_schema(db)
    async with db.connect() as con:
        assert (await (await con.execute('PRAGMA user_version')).fetchone())[0] == 2
        assert (await (await con.execute('SELECT identifier FROM users')).fetchone())[0] == 'old'
