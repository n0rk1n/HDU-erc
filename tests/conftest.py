from __future__ import annotations

import pytest_asyncio

from chatbot.db.connection import Database
from chatbot.db.schema import initialize_schema


@pytest_asyncio.fixture
async def database(tmp_path):
    database = Database(tmp_path / "chatbot.sqlite3")
    await initialize_schema(database)
    return database
