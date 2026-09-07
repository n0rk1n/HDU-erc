"""Validate historical references separately from current-turn analyses."""
import json
from chatbot.core.errors import InvalidMessageState


async def build_reply_emotion_context(*, gate, decision_id, analysis_id,
                                     conversation_id, user_message_id):
    decision = await gate.repository.get(decision_id)
    if (decision is None or decision.status != 'completed' or decision.conversation_id != conversation_id
            or decision.user_message_id != user_message_id):
        raise InvalidMessageState('gate decision identity mismatch')
    policy = decision.snapshot['policy']
    current_turn = decision.snapshot['facts']['current_turn']
    if decision.action == 'analyze':
        if not analysis_id or decision.analysis_id != analysis_id:
            raise InvalidMessageState('gate analysis missing or mismatched')
        analysis = await gate.emotions.get(analysis_id)
        if (analysis is None or analysis.user_message_id != user_message_id
                or analysis.conversation_id != conversation_id):
            raise InvalidMessageState('emotion analysis identity mismatch')
        if analysis.status == 'completed':
            return json.dumps({'result': analysis.result_json, 'source_analysis_id': analysis.id,
                'source_user_message_id': user_message_id, 'source_turn': current_turn,
                'elapsed_turns': 0, 'is_historical': False}, ensure_ascii=False)
        if analysis.status != 'failed':
            raise InvalidMessageState('emotion analysis is not terminal')
        reuse = policy['reuse_last_success_on_analysis_failure']
    else:
        if analysis_id:
            raise InvalidMessageState('skipped gate cannot have current analysis')
        reuse = policy['reuse_last_success_on_skip']
    baseline = decision.snapshot.get('baseline')
    if not reuse or not baseline:
        return None
    analysis = await gate.emotions.get(baseline['analysis_id'])
    if (analysis is None or analysis.status != 'completed' or analysis.conversation_id != conversation_id
            or analysis.user_message_id != baseline['user_message_id']):
        raise InvalidMessageState('historical emotion identity mismatch')
    current = await gate.messages.find_turn(conversation_id, decision.request_id)
    source = await gate.messages.find_turn(conversation_id, analysis.request_id)
    if source is None or current is None or source.user.sequence_no >= current.user.sequence_no:
        raise InvalidMessageState('historical emotion must precede current input')
    source_facts = await gate.repository.facts(conversation_id, analysis.user_message_id)
    if source_facts.current_turn != baseline['source_turn']:
        raise InvalidMessageState('historical emotion turn mismatch')
    return json.dumps({'result': analysis.result_json, 'source_analysis_id': analysis.id,
        'source_user_message_id': analysis.user_message_id, 'source_turn': source_facts.current_turn,
        'elapsed_turns': current_turn - source_facts.current_turn, 'is_historical': True}, ensure_ascii=False)
