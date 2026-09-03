from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
import pytest_asyncio

from chatbot.core.errors import InvalidIdentifier
from chatbot.db.conversations import ConversationRepository
from chatbot.services.identity import IdentityService


@pytest_asyncio.fixture
async def identity_service(database) -> IdentityService:
    return IdentityService(database)


@pytest.mark.asyncio
async def test_resolve_reuses_trimmed_identifier(identity_service: IdentityService) -> None:
    """Catches identity partitioning that treats harmless outer whitespace as a new user."""
    first = await identity_service.resolve("  Alice  ")
    second = await identity_service.resolve("Alice")

    assert first.user.id == second.user.id
    assert first.user.identifier == "Alice"
    assert first.conversation.thread_id == second.conversation.thread_id


@pytest.mark.asyncio
async def test_identifier_is_case_sensitive(identity_service: IdentityService) -> None:
    """Catches identity normalization that lowercases distinct user partitions."""
    upper, lower = await asyncio.gather(
        identity_service.resolve("Alice"),
        identity_service.resolve("alice"),
    )

    assert upper.user.id != lower.user.id
    assert upper.user.identifier == "Alice"
    assert lower.user.identifier == "alice"


@pytest.mark.asyncio
async def test_concurrent_resolve_creates_one_default(identity_service: IdentityService, database) -> None:
    """Catches unprotected create races that duplicate a shared user's default conversation."""
    results = await asyncio.gather(
        *(identity_service.resolve("shared") for _ in range(8))
    )

    assert len({item.user.id for item in results}) == 1
    assert len({item.conversation.id for item in results}) == 1

    async with database.connect() as connection:
        user_count = (
            await (await connection.execute("SELECT count(*) FROM users WHERE identifier = 'shared'")).fetchone()
        )[0]
        default_count = (
            await (
                await connection.execute(
                    "SELECT count(*) FROM conversations WHERE user_id = ? AND is_default = 1",
                    (results[0].user.id,),
                )
            ).fetchone()
        )[0]

    assert user_count == 1
    assert default_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["   ", "x" * 129, None])
async def test_resolve_rejects_invalid_identifiers_with_stable_domain_error(
    identity_service: IdentityService, identifier: object
) -> None:
    """Catches invalid external identifiers reaching SQLite or leaking unstable validation failures."""
    with pytest.raises(InvalidIdentifier) as error:
        await identity_service.resolve(identifier)  # type: ignore[arg-type]

    assert error.value.code == "invalid_identifier"
    assert error.value.http_status == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["\x00", "alice\x00example"])
async def test_resolve_rejects_nul_identifiers_without_creating_records(
    identity_service: IdentityService, database, identifier: str
) -> None:
    """Catches NUL values bypassing Python length checks and leaking SQLite constraint errors."""
    with pytest.raises(InvalidIdentifier) as error:
        await identity_service.resolve(identifier)

    assert error.value.code == "invalid_identifier"
    assert error.value.http_status == 422
    async with database.connect() as connection:
        user_count = (await (await connection.execute("SELECT count(*) FROM users")).fetchone())[0]
        conversation_count = (
            await (await connection.execute("SELECT count(*) FROM conversations")).fetchone()
        )[0]

    assert user_count == 0
    assert conversation_count == 0


@pytest.mark.asyncio
async def test_default_conversation_uses_distinct_uuid4_ids_and_keeps_future_nondefaults(
    identity_service: IdentityService, database
) -> None:
    """Catches thread/business ID reuse or a schema that prevents later non-default conversations."""
    resolved = await identity_service.resolve("future-conversations")

    assert UUID(resolved.conversation.id).version == 4
    assert UUID(resolved.conversation.thread_id).version == 4
    assert resolved.conversation.id != resolved.conversation.thread_id

    async with database.transaction(immediate=True) as connection:
        await connection.execute(
            """
            INSERT INTO conversations(
                id, user_id, thread_id, is_default, title, status, created_at, updated_at
            ) VALUES (?, ?, ?, 0, 'Later', 'active', ?, ?)
            """,
            (
                "b0ec8385-7ea8-4d97-b7e1-02c2817f92cd",
                resolved.user.id,
                "48f2c3ca-36dd-47c5-b90d-ecb869ec0859",
                "2026-09-03T00:00:00+00:00",
                "2026-09-03T00:00:00+00:00",
            ),
        )

    default = await ConversationRepository(database).get_default_by_user(resolved.user.id)

    assert default == resolved.conversation
