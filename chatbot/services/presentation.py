"""Allowlisted UI projections of durable facts; never publish internal model traces."""
from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from chatbot.db.connection import Database


EMOTION_NAMES = {
    'surprised': '惊讶', 'excited': '兴奋', 'annoyed': '烦恼', 'proud': '自豪',
    'angry': '生气', 'sad': '难过', 'grateful': '感激', 'lonely': '孤独',
    'impressed': '赞叹', 'afraid': '害怕', 'disgusted': '厌恶', 'confident': '自信',
    'terrified': '恐惧', 'hopeful': '充满希望', 'anxious': '焦虑', 'disappointed': '失望',
    'joyful': '喜悦', 'prepared': '准备充分', 'guilty': '内疚', 'furious': '愤怒',
    'nostalgic': '怀念', 'jealous': '嫉妒', 'anticipating': '期待', 'embarrassed': '尴尬',
    'content': '满足', 'devastated': '悲痛', 'sentimental': '感怀', 'caring': '关切',
    'trusting': '信任', 'ashamed': '羞愧', 'apprehensive': '忐忑', 'faithful': '忠诚',
    'neutral': '情绪平稳', 'no_emotion': '未表达明确情绪',
}


def _emotion(row):
    if row['analysis_status'] != 'completed' or not row['result_json']:
        return None
    result = json.loads(row['result_json'])
    label = result['primary_emotion']
    return {
        'label': label, 'display_label': EMOTION_NAMES.get(label, label),
        'confidence': result['confidence'], 'evidence': result['evidence'],
        'source_message_id': row['user_message_id'], 'sequence_no': row['user_sequence'],
        'analyzed_at': row['analysis_completed_at'],
    }


def _elapsed(row):
    if not row['completed_at']:
        return None
    try:
        return max(0, int((datetime.fromisoformat(row['completed_at']) -
                           datetime.fromisoformat(row['created_at'])).total_seconds() * 1000))
    except (TypeError, ValueError):
        return None


def _processing(row):
    terminal = row['status'] in ('completed', 'failed')
    if terminal and row['gate_status'] is None and row['analysis_status'] is None:
        return None  # Legacy turns have no recorded processing facts.
    decision = row['gate_status']
    decision = decision if decision in ('completed', 'failed') else ('failed' if terminal else 'running')
    analysis = row['analysis_status']
    legacy_analysis = row['gate_status'] is None and analysis is not None
    if analysis in ('completed', 'failed'):
        emotion_status = analysis
    elif row['gate_action'] == 'skip':
        emotion_status = 'skipped'
    elif analysis or row['gate_action'] == 'analyze':
        emotion_status = 'failed' if terminal else 'running'
    else:
        emotion_status = 'not_started'
    can_reply = emotion_status in ('completed', 'failed', 'skipped') and (decision == 'completed' or legacy_analysis)
    response = row['status'] if terminal else ('running' if can_reply else 'pending')
    if terminal and not can_reply:
        response = 'skipped'
    steps = [{'id': 'received', 'status': 'completed'}]
    if not legacy_analysis:
        steps.append({'id': 'decision', 'status': decision})
    # Before the decision there is no claim that recognition will be invoked.
    if emotion_status != 'not_started':
        steps.append({'id': 'emotion', 'status': emotion_status})
    steps.append({'id': 'response', 'status': response})
    return {
        'steps': steps, 'emotion_status': emotion_status,
        'emotion_invoked': row['analysis_started_at'] is not None,
        'emotion': _emotion(row), 'elapsed_ms': _elapsed(row),
        'started_at': row['created_at'],
    }


class PresentationService:
    def __init__(self, database: Database):
        self.database = database

    async def snapshot(self, conversation_id: str, request_ids: list[str]):
        """Batch visible turns, plus latest success across the entire conversation."""
        processing = {}
        async with self.database.connect() as con:
            con.row_factory = aiosqlite.Row
            # Keep the two projections on one SQLite read snapshot during generation.
            await con.execute('BEGIN')
            if request_ids:
                placeholders = ','.join('?' for _ in request_ids)
                rows = await (await con.execute(f'''
                    SELECT m.request_id,m.status,m.created_at,m.completed_at,
                        u.id AS user_message_id,u.sequence_no AS user_sequence,
                        g.status AS gate_status,g.action AS gate_action,
                        a.status AS analysis_status,a.result_json,
                        a.started_at AS analysis_started_at,a.completed_at AS analysis_completed_at
                    FROM messages m
                    JOIN messages u ON u.conversation_id=m.conversation_id
                        AND u.request_id=m.request_id AND u.role='user'
                    LEFT JOIN emotion_gate_decisions g ON g.user_message_id=u.id
                    LEFT JOIN emotion_analyses a ON a.user_message_id=u.id
                    WHERE m.conversation_id=? AND m.role='assistant'
                        AND m.request_id IN ({placeholders})
                ''', (conversation_id, *request_ids))).fetchall()
                processing = {row['request_id']: _processing(row) for row in rows}
            latest = await (await con.execute('''
                SELECT a.status AS analysis_status,a.result_json,a.user_message_id,
                    u.sequence_no AS user_sequence,a.completed_at AS analysis_completed_at
                FROM emotion_analyses a JOIN messages u ON u.id=a.user_message_id
                WHERE a.conversation_id=? AND a.status='completed'
                ORDER BY u.sequence_no DESC LIMIT 1
            ''', (conversation_id,))).fetchone()
        return processing, _emotion(latest) if latest else None

    async def turn(self, conversation_id: str, request_id: str):
        processing, latest = await self.snapshot(conversation_id, [request_id])
        return {'processing': processing.get(request_id), 'latest_emotion': latest}
