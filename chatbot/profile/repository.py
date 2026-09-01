"""Client-scoped profile persistence with temporary synchronous compatibility."""

from __future__ import annotations

from typing import Any, Coroutine

from langgraph.store.base import BaseStore

from chatbot.profile.onboarding import sanitize_profile
from chatbot.core.runtime_store import DEFAULT_RUNTIME_DB_PATH, RuntimeStore

RUNTIME_DB_PATH = DEFAULT_RUNTIME_DB_PATH


async def _load_profile(store: BaseStore, client_id: str) -> dict[str, str]:
    item = await store.aget((client_id, "profile"), "current")
    if item is None:
        return {}
    return {key: value for key, value in item.value.items() if isinstance(value, str) and value}


async def _save_profile(store: BaseStore, client_id: str, profile: dict[str, Any]) -> None:
    await store.aput((client_id, "profile"), "current", sanitize_profile(profile))


def load_profile(
    store: BaseStore | None = None,
    client_id: str | None = None,
) -> dict[str, str] | Coroutine[Any, Any, dict[str, str]]:
    """Load a profile from the Store, or use the legacy runtime DB without arguments."""
    if store is None and client_id is None:
        return RuntimeStore(RUNTIME_DB_PATH).load_profile()
    if store is None or client_id is None:
        raise TypeError("load_profile requires both store and client_id.")
    return _load_profile(store, client_id)


def save_profile(
    store_or_profile: BaseStore | dict[str, Any],
    client_id: str | None = None,
    profile: dict[str, Any] | None = None,
) -> bool | Coroutine[Any, Any, None]:
    """Save to the Store, with a no-argument-shape legacy compatibility adapter."""
    if isinstance(store_or_profile, dict) and client_id is None and profile is None:
        return RuntimeStore(RUNTIME_DB_PATH).replace_profile(sanitize_profile(store_or_profile))
    if client_id is None or profile is None:
        raise TypeError("save_profile requires store, client_id, and profile.")
    return _save_profile(store_or_profile, client_id, profile)


def format_profile(profile: dict[str, str]) -> str:
    if not profile:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in profile.items())
