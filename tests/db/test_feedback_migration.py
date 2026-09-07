from chatbot.db.connection import Database
from chatbot.db.schema import V1_DDL_STATEMENTS, V2_DDL_STATEMENTS, V3_DDL_STATEMENTS, V4_DDL_STATEMENTS, initialize_schema
from chatbot.services.identity import IdentityService


async def test_upgrade_v4_preserves_all_message_facts_and_defaults_feedback(tmp_path):
    """Catches destructive migration, lost bubble/audit data, or fabricated ratings."""
    database = Database(tmp_path / 'v4.sqlite3')
    async with database.transaction() as connection:
        for ddl in (*V1_DDL_STATEMENTS, *V2_DDL_STATEMENTS, *V3_DDL_STATEMENTS, *V4_DDL_STATEMENTS):
            await connection.execute(ddl)
        await connection.execute('PRAGMA user_version=4')
    conversation = (await IdentityService(database).resolve('legacy')).conversation
    async with database.transaction() as connection:
        await connection.execute('''INSERT INTO messages(id,conversation_id,request_id,sequence_no,
            role,status,content,trace_json,bubbles_json,created_at,updated_at)
            VALUES ('old',?,'r',1,'assistant','completed','正文','{"audit":"keep"}','["正文"]','then','then')''', (conversation.id,))
        before = await (await connection.execute('SELECT * FROM messages')).fetchone()
    await initialize_schema(database)
    await initialize_schema(database)
    async with database.connect() as connection:
        columns = [r[1] for r in await (await connection.execute('PRAGMA table_info(messages)')).fetchall()]
        assert 'feedback' in columns
        after = await (await connection.execute('SELECT * FROM messages')).fetchone()
        assert after[:len(before)] == before
        assert after[columns.index('feedback')] is None
        assert await (await connection.execute('PRAGMA foreign_key_check')).fetchall() == []


async def test_concurrent_ratings_keep_only_first_vote(database):
    """Catches check-then-write races accepting two ratings for one reply."""
    import asyncio
    from chatbot.core.errors import AlreadyRated
    from chatbot.db.messages import MessageRepository

    conversation = (await IdentityService(database).resolve('voter')).conversation
    repository = MessageRepository(database)
    turn = await repository.reserve_turn(conversation.id, 'rating-race', '你好')
    async with database.transaction() as connection:
        await connection.execute("UPDATE messages SET status='completed', content='回复' WHERE id=?", (turn.assistant.id,))
    results = await asyncio.gather(
        repository.set_feedback(conversation.id, turn.assistant.id, 'like'),
        repository.set_feedback(conversation.id, turn.assistant.id, 'dislike'),
        return_exceptions=True,
    )
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, AlreadyRated) for result in results) == 1
    saved = await repository.find_assistant(conversation.id, 'rating-race')
    assert saved.feedback == ('like' if results[0] is None else 'dislike')
