from chatbot.db.messages import MessageRepository
from chatbot.services.identity import IdentityService

async def test_history_pairs_by_request_and_excludes_failed_future_and_other_conversation(database):
    from chatbot.emotion.history import group_history
    repo=MessageRepository(database)
    convo=(await IdentityService(database).resolve('history')).conversation
    old=await repo.reserve_turn(convo.id,'old','old human')
    await repo.mark_streaming(old.assistant.id)
    await repo.complete_assistant(old.assistant.id,content='old assistant',reasoning_content=None,trace={},prompt=[],provider=None,model=None,parameters=None,input_tokens=None,output_tokens=None,total_tokens=None,latency_ms=None,finish_reason=None)
    failed=await repo.reserve_turn(convo.id,'failed','failed human')
    await repo.mark_streaming(failed.assistant.id)
    await repo.fail_assistant(failed.assistant.id,error_code='model_error',error_message='failed')
    current=await repo.reserve_turn(convo.id,'current','current human')
    await repo.reserve_turn(convo.id,'future','future human')
    rows=await repo.list_emotion_history(convo.id,through_sequence=current.user.sequence_no)
    turns,human,excluded=group_history(rows,current_id=current.user.id)
    assert len(turns)==1
    assert [m.type for m in turns[0].messages]==['human','ai']
    assert human.content=='current human'
    assert failed.user.id in excluded and failed.assistant.id in excluded
