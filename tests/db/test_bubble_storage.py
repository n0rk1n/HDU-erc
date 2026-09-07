import json

from chatbot.db.connection import Database
from chatbot.db.messages import MessageRepository
from chatbot.db.schema import V1_DDL_STATEMENTS, V2_DDL_STATEMENTS, V3_DDL_STATEMENTS, initialize_schema
from chatbot.services.identity import IdentityService


async def test_upgrade_v3_preserves_legacy_message_text_and_audit(tmp_path):
    """Catches destructive migration or reinterpreting old paragraphs as bubble boundaries."""
    database = Database(tmp_path / 'v3.sqlite3')
    async with database.transaction() as connection:
        for ddl in (*V1_DDL_STATEMENTS, *V2_DDL_STATEMENTS, *V3_DDL_STATEMENTS):
            await connection.execute(ddl)
        await connection.execute('PRAGMA user_version=3')
    conversation = (await IdentityService(database).resolve('legacy')).conversation
    async with database.transaction() as connection:
        await connection.execute('''INSERT INTO messages(id,conversation_id,request_id,sequence_no,
            role,status,content,trace_json,reasoning_content,created_at,updated_at)
            VALUES ('old',?,'r',1,'assistant','completed',?,'{"audit":"keep"}','kept','2026-09-03','2026-09-03')''',
            (conversation.id, '旧正文。\n\n还是同一条。'))
        before = await (await connection.execute('SELECT * FROM messages')).fetchone()
    await initialize_schema(database)
    await initialize_schema(database)
    async with database.connect() as connection:
        after = await (await connection.execute('SELECT * FROM messages')).fetchone()
        assert after[:-1] == before
        assert after[-1] is None
        assert await (await connection.execute('PRAGMA foreign_key_check')).fetchall() == []
    history = await MessageRepository(database).list_visible(conversation.id, limit=10)
    assert history[0]['content'] == '旧正文。\n\n还是同一条。'
    assert history[0]['bubbles'] is None


async def test_recovery_keeps_saved_bubbles_after_process_interruption(database):
    """Catches a restart merging or dropping already published complete bubbles."""
    conversation = (await IdentityService(database).resolve('restart')).conversation
    messages = MessageRepository(database)
    turn = await messages.reserve_turn(conversation.id, 'r', '继续聊')
    await messages.mark_streaming(turn.assistant.id)
    await messages.save_bubbles(turn.assistant.id, ['先说完这句。', '再说完一句。'])
    restarted = MessageRepository(Database(database.path))
    await restarted.fail_interrupted(error_code='process_interrupted', stale_before='9999')
    saved = await restarted.find_assistant(conversation.id, 'r')
    assert saved.status == 'failed'
    assert saved.content == '先说完这句。\n\n再说完一句。'
    assert json.loads(saved.bubbles_json) == ['先说完这句。', '再说完一句。']
