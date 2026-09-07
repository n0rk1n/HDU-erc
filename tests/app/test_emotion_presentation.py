"""Public progress must reflect durable facts, survive replay and isolate users."""
import json

import pytest
from uuid import uuid4

from fastapi.testclient import TestClient

from chatbot.db.connection import Database
from chatbot.emotion.types import ModelOutcome
from chatbot.emotion_gate.types import GatePolicy
from chatbot.llm.types import TokenUsage
from tests.api.conftest import app_config
from tests.api.helpers import OfflineModel, parse_sse
from tests.emotion.helpers import create_app, emotion_runtime
from tests.emotion_gate.helpers import gate_runtime


class Recognizer:
    parameters = {'model': 'offline', 'max_retries': 0}

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    async def invoke(self, messages):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('private provider failure sk-secret')
        return ModelOutcome(json.dumps({
            'primary_emotion': 'sad', 'confidence': .9, 'secondary_emotions': ['lonely'],
            'evidence': '你提到最近感到难过。', 'reply_strategy': 'private strategy',
            'trajectory_note': 'private trajectory', 'safety_level': 'normal',
        }), 'private reasoning', TokenUsage(), 'stop', {}, None)


def resolve(client, name):
    return client.post('/api/users/resolve', json={'identifier': name}).json()['user']['id']


def send(client, user, request_id=None):
    request_id = request_id or str(uuid4())
    response = client.post(f'/api/users/{user}/messages:stream',
                           json={'request_id': request_id, 'content': '最近感到难过。'})
    assert response.status_code == 200
    return parse_sse(response.text)


def test_progress_success_skip_history_pagination_replay_and_user_isolation(app_config):
    db = Database(app_config.sqlite_db_path)
    recognizer = Recognizer()
    app = create_app(config=app_config, model=OfflineModel(),
                     emotion=emotion_runtime(db, recognizer), gate=gate_runtime(db))
    with TestClient(app) as client:
        alice, bob = resolve(client, 'alice'), resolve(client, 'bob')
        request = str(uuid4())
        events = send(client, alice, request)
        progress = [data for name, data in events if name == 'progress']
        assert progress, 'The stream must report real processing before the first reply token'
        states = [(s['id'], s['status']) for p in progress for s in p['processing']['steps']]
        assert ('decision', 'running') in states
        assert ('emotion', 'running') in states
        assert ('emotion', 'completed') in states
        first_token = next(i for i, (name, _) in enumerate(events) if name == 'token')
        assert all(i < first_token for i, (name, _) in enumerate(events) if name == 'progress')
        done = events[-1][1]
        result = done['message']['processing']['emotion']
        assert result['label'] == 'sad' and result['display_label'] == '难过'
        assert result['confidence'] == .9 and result['evidence'] == '你提到最近感到难过。'
        assert result['sequence_no'] == 1
        assert done['message']['processing']['emotion_status'] == 'completed'
        assert done['message']['processing']['elapsed_ms'] >= 0
        assert done['latest_emotion'] == result
        for secret in ('private reasoning', 'private strategy', 'private trajectory', 'sk-secret',
                       'snapshot_json', 'thread_id', 'raw_output'):
            assert secret not in json.dumps(events)
        replay = send(client, alice, request)
        assert replay[-1][1]['replayed'] is True
        assert replay[-1][1]['message']['processing'] == done['message']['processing']
        assert not any(name == 'progress' for name, _ in replay)
        skipped = send(client, alice)[-1][1]
        assert skipped['message']['processing']['emotion_status'] == 'skipped'
        assert skipped['message']['processing']['emotion'] is None
        assert skipped['latest_emotion'] == result
        assert recognizer.calls == 1
        history = client.get(f'/api/users/{alice}/messages?limit=1').json()
        assert history['latest_emotion'] == result  # Not limited to the visible message page.
        assert history['messages'][0]['processing'] == skipped['message']['processing']
        empty = client.get(f'/api/users/{bob}/messages').json()
        assert empty['latest_emotion'] is None and empty['messages'] == []


def test_failed_analysis_keeps_previous_badge_but_never_claims_new_success(app_config):
    db = Database(app_config.sqlite_db_path)
    model = Recognizer(fail_at=2)
    with TestClient(create_app(config=app_config, model=OfflineModel(),
            emotion=emotion_runtime(db, model),
            gate=gate_runtime(db, policy=GatePolicy(max_interval_turns=1)))) as client:
        user = resolve(client, 'failure')
        previous = send(client, user)[-1][1]['latest_emotion']
        events = send(client, user)
        assert events[-1][0] == 'done'
        failed = events[-1][1]['message']['processing']
        assert failed['emotion_status'] == 'failed' and failed['emotion'] is None
        assert failed['emotion_invoked'] is True
        assert events[-1][1]['latest_emotion'] == previous
        assert 'sk-secret' not in json.dumps(events)
        history = client.get(f'/api/users/{user}/messages').json()
        assert history['messages'][-1]['processing'] == failed
        assert history['latest_emotion'] == previous


def test_reply_failure_retains_successful_emotion_in_history(app_config):
    db = Database(app_config.sqlite_db_path)
    with TestClient(create_app(config=app_config, model=OfflineModel(fail=True),
            emotion=emotion_runtime(db, Recognizer()), gate=gate_runtime(db))) as client:
        user = resolve(client, 'reply-failure')
        assert send(client, user)[-1][0] == 'error'
        history = client.get(f'/api/users/{user}/messages').json()
        processing = history['messages'][-1]['processing']
        assert processing['emotion_status'] == 'completed'
        assert processing['steps'][-1] == {'id': 'response', 'status': 'failed'}
        assert history['latest_emotion']['label'] == 'sad'


def test_preparation_failure_is_not_reported_as_a_model_call(app_config):
    from dataclasses import replace
    from chatbot.emotion.types import BudgetConfig
    db = Database(app_config.sqlite_db_path)
    model = Recognizer()
    emotion = replace(emotion_runtime(db, model), budget=BudgetConfig(30, 10, 5))
    with TestClient(create_app(config=app_config, model=OfflineModel(), emotion=emotion,
                              gate=gate_runtime(db))) as client:
        user = resolve(client, 'budget-failure')
        done = send(client, user)[-1][1]
        assert done['message']['processing']['emotion_status'] == 'failed'
        assert done['message']['processing']['emotion_invoked'] is False
        assert done['latest_emotion'] is None and model.calls == 0


def test_history_reports_running_recognition_before_reply_exists(app_config):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()

    class PausedRecognizer(Recognizer):
        async def invoke(self, messages):
            entered.set()
            await asyncio.to_thread(release.wait, 10)
            return await super().invoke(messages)

    db = Database(app_config.sqlite_db_path)
    with TestClient(create_app(config=app_config, model=OfflineModel(),
            emotion=emotion_runtime(db, PausedRecognizer()), gate=gate_runtime(db))) as client:
        user = resolve(client, 'running-recognition')
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(send, client, user)
            try:
                assert entered.wait(5)
                history = client.get(f'/api/users/{user}/messages').json()
                processing = history['messages'][-1]['processing']
                assert processing['emotion_status'] == 'running'
                assert processing['emotion_invoked'] is True
                assert processing['steps'][-1] == {'id': 'response', 'status': 'pending'}
                assert processing['emotion'] is None and history['latest_emotion'] is None
            finally:
                release.set()
            assert pending.result(timeout=5)[-1][0] == 'done'


def test_older_analysis_without_gate_does_not_invent_a_failed_decision(app_config):
    import sqlite3
    db = Database(app_config.sqlite_db_path)
    with TestClient(create_app(config=app_config, model=OfflineModel(),
            emotion=emotion_runtime(db, Recognizer()), gate=gate_runtime(db))) as client:
        user = resolve(client, 'pre-gate-history')
        send(client, user)
        # Represents an analysis recorded before the gate was introduced.
        with sqlite3.connect(db.path) as con:
            con.execute('DELETE FROM emotion_gate_decisions')
        processing = client.get(f'/api/users/{user}/messages').json()['messages'][-1]['processing']
        assert processing['emotion_status'] == 'completed'
        assert {'id': 'response', 'status': 'completed'} in processing['steps']
        assert not any(step['id'] == 'decision' for step in processing['steps'])


@pytest.mark.parametrize(('names', 'expected'), [({'sad': '低落'}, '低落'), ({}, 'sad')])
def test_configured_names_reach_progress_done_replay_and_history(app_config, tmp_path, monkeypatch, names, expected):
    names_path = tmp_path / 'names.json'
    names_path.write_text(json.dumps(names), encoding='utf-8')
    monkeypatch.setenv('EMOTION_NAMES_PATH', str(names_path))
    db = Database(app_config.sqlite_db_path)
    with TestClient(create_app(config=app_config, model=OfflineModel(),
            emotion=emotion_runtime(db, Recognizer()), gate=gate_runtime(db))) as client:
        user = resolve(client, 'custom-names')
        request_id = str(uuid4())
        events = send(client, user, request_id)
        completed_progress = [data for name, data in events if name == 'progress'
                              and data['processing']['emotion']]
        assert completed_progress
        for data in completed_progress:
            assert data['processing']['emotion']['display_label'] == expected
            assert data['latest_emotion']['display_label'] == expected
        for result in (events[-1][1], send(client, user, request_id)[-1][1]):
            assert result['message']['processing']['emotion']['display_label'] == expected
            assert result['latest_emotion']['label'] == 'sad'
            assert result['latest_emotion']['display_label'] == expected
        history = client.get(f'/api/users/{user}/messages').json()
        assert history['messages'][-1]['processing']['emotion']['display_label'] == expected
        assert history['latest_emotion']['display_label'] == expected
        names_path.write_text('{"sad": "伤心"}', encoding='utf-8')
        assert client.get(f'/api/users/{user}/messages').json()['latest_emotion']['display_label'] == '伤心'


@pytest.mark.parametrize('content', ['[]', '{', '{"sad": 1}', '{"sad": " "}',
                                     '{" ": "伤心"}', '{"sad":"a","sad":"b"}', None])
def test_invalid_emotion_names_are_rejected(tmp_path, monkeypatch, content):
    from chatbot.core.errors import ConfigError
    from chatbot.services.presentation import PresentationService
    names_path = tmp_path / 'names.json'
    if content is not None:
        names_path.write_text(content, encoding='utf-8')
    monkeypatch.setenv('EMOTION_NAMES_PATH', str(names_path))
    with pytest.raises(ConfigError):
        PresentationService(Database(tmp_path / 'unused.sqlite3'))
