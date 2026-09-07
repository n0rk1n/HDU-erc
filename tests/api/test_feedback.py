import sqlite3
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tests.api.helpers import parse_sse, OfflineModel
from tests.emotion.helpers import create_app
from tests.api.test_messages import _complete_turn


def test_feedback_persists_and_replays_without_changing_reply(client, resolved_user, app_config):
    """Catches lost feedback on history/restart/replay or rating overwriting reply facts."""
    user_id = resolved_user['user']['id']
    request_id = _complete_turn(client, user_id)
    history_url = f'/api/users/{user_id}/messages'
    original = client.get(history_url).json()['messages'][-1]
    url = f"{history_url}/{original['id']}/feedback"
    with sqlite3.connect(app_config.sqlite_db_path) as connection:
        before = connection.execute('SELECT * FROM messages ORDER BY sequence_no').fetchall()
    rating = "dislike"
    response = client.patch(url, json={'feedback': rating})
    assert response.status_code == 200
    assert response.json() == {'message_id': original['id'], 'feedback': rating}
    saved = client.get(history_url).json()['messages'][-1]
    assert saved['feedback'] == rating
    assert {k: v for k, v in saved.items() if k != 'feedback'} == {k: v for k, v in original.items() if k != 'feedback'}
    for rating in ('dislike', 'like'):
        repeated = client.patch(url, json={'feedback': rating})
        assert repeated.status_code == 409
        assert repeated.json()['error']['code'] == 'already_rated'
    with TestClient(create_app(config=app_config, model=OfflineModel())) as restarted:
        assert restarted.get(history_url).json()['messages'][-1]['feedback'] == 'dislike'
        replay = restarted.post(history_url + ':stream', json={'request_id': request_id, 'content': '你好'})
        assert parse_sse(replay.text)[-1][1]['message']['feedback'] == 'dislike'
    with sqlite3.connect(app_config.sqlite_db_path) as connection:
        after = connection.execute('SELECT * FROM messages ORDER BY sequence_no').fetchall()
    assert [row[:-1] for row in after] == [row[:-1] for row in before]


def test_feedback_ownership_and_missing_message(client, resolved_user):
    """Catches updating another user's message using its globally unique ID."""
    user_id = resolved_user['user']['id']
    _complete_turn(client, user_id)
    message = client.get(f'/api/users/{user_id}/messages').json()['messages'][-1]
    other = client.post('/api/users/resolve', json={'identifier': 'Bob'}).json()['user']['id']
    for owner, message_id in ((other, message['id']), (user_id, str(uuid4()))):
        response = client.patch(f'/api/users/{owner}/messages/{message_id}/feedback', json={'feedback': 'like'})
        assert response.status_code == 404
        assert response.json()['error']['code'] == 'message_not_found'
    assert client.get(f'/api/users/{user_id}/messages').json()['messages'][-1]['feedback'] is None


@pytest.mark.parametrize('status', ['pending', 'streaming', 'failed', 'user'])
def test_feedback_only_accepts_completed_assistant(client, resolved_user, app_config, status):
    """Catches ratings on human input, incomplete output, or failed replies."""
    user_id = resolved_user['user']['id']
    _complete_turn(client, user_id)
    messages = client.get(f'/api/users/{user_id}/messages').json()['messages']
    message_id = messages[0 if status == 'user' else 1]['id']
    if status != 'user':
        with sqlite3.connect(app_config.sqlite_db_path) as connection:
            connection.execute('UPDATE messages SET status=? WHERE id=?', (status, message_id))
    response = client.patch(f'/api/users/{user_id}/messages/{message_id}/feedback', json={'feedback': 'like'})
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'invalid_identifier_or_message'


@pytest.mark.parametrize('payload', [{'feedback': 'neutral'}, {'feedback': None}, {}, {'feedback': 1}])
def test_feedback_rejects_invalid_values(client, resolved_user, payload):
    user_id = resolved_user['user']['id']
    _complete_turn(client, user_id)
    message = client.get(f'/api/users/{user_id}/messages').json()['messages'][-1]
    response = client.patch(f"/api/users/{user_id}/messages/{message['id']}/feedback", json=payload)
    assert response.status_code == 400


def test_failed_feedback_write_does_not_report_success_and_can_retry(client, resolved_user, app_config):
    """Catches a failed DB transaction leaving a false saved rating or blocking retry."""
    user_id = resolved_user['user']['id']
    _complete_turn(client, user_id)
    history_url = f'/api/users/{user_id}/messages'
    message = client.get(history_url).json()['messages'][-1]
    url = f"{history_url}/{message['id']}/feedback"
    with sqlite3.connect(app_config.sqlite_db_path) as connection:
        connection.execute("CREATE TRIGGER reject_rating BEFORE UPDATE OF feedback ON messages BEGIN SELECT RAISE(ABORT, 'offline-write-failure'); END")
    response = TestClient(client.app, raise_server_exceptions=False).patch(url, json={'feedback': 'like'})
    assert response.status_code == 500
    assert 'offline-write-failure' not in response.text
    assert client.get(history_url).json()['messages'][-1]['feedback'] is None
    with sqlite3.connect(app_config.sqlite_db_path) as connection:
        connection.execute('DROP TRIGGER reject_rating')
    assert client.patch(url, json={'feedback': 'like'}).status_code == 200
    assert client.get(history_url).json()['messages'][-1]['feedback'] == 'like'
