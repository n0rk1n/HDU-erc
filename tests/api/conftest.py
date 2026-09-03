from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from chatbot.core.config import AppConfig
from chatbot.web import create_app
from tests.api.helpers import OfflineModel


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    return AppConfig(
        llm_api_key=SecretStr("sk-test-secret"),
        llm_model="offline-test",
        llm_base_url=None,
        llm_temperature=0,
        llm_timeout_seconds=1,
        context_message_limit=20,
        sqlite_db_path=tmp_path / "chatbot.sqlite3",
    )


@pytest.fixture
def offline_model() -> OfflineModel:
    return OfflineModel()


@pytest.fixture
def app(app_config, offline_model):
    return create_app(config=app_config, model=offline_model)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def resolved_user(client: TestClient) -> dict[str, object]:
    response = client.post("/api/users/resolve", json={"identifier": "Alice"})
    assert response.status_code == 200
    return response.json()
