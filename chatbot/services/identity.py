from __future__ import annotations

from dataclasses import dataclass

from chatbot.core.errors import InvalidIdentifier
from chatbot.db.connection import Database
from chatbot.db.conversations import ConversationRepository
from chatbot.db.models import Conversation, User
from chatbot.db.users import UserRepository


def normalize_identifier(identifier: str) -> str:
    """Validate an external user identifier without altering its case."""
    if not isinstance(identifier, str):
        raise InvalidIdentifier()
    normalized = identifier.strip()
    if "\x00" in normalized or not 1 <= len(normalized) <= 128:
        raise InvalidIdentifier()
    return normalized


@dataclass(frozen=True)
class ResolvedIdentity:
    user: User
    conversation: Conversation


class IdentityService:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.users = UserRepository()
        self.conversations = ConversationRepository(database)

    async def resolve(self, identifier: str) -> ResolvedIdentity:
        normalized = normalize_identifier(identifier)
        async with self.database.transaction(immediate=True) as connection:
            user = await self.users.get_or_create(connection, normalized)
            conversation = await self.conversations.get_or_create_default(connection, user.id)
        return ResolvedIdentity(user=user, conversation=conversation)
