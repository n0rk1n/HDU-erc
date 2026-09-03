"""Client-scoped profile persistence in the LangGraph Store."""

from __future__ import annotations

from typing import Any

from langgraph.store.base import BaseStore

from chatbot.profile.onboarding import sanitize_profile


async def load_profile(store: BaseStore, client_id: str) -> dict[str, str]:
    item = await store.aget((client_id, "profile"), "current")
    if item is None:
        return {}
    return {key: value for key, value in item.value.items() if isinstance(value, str) and value}


async def save_profile(
    store: BaseStore,
    client_id: str,
    profile: dict[str, Any],
) -> None:
    await store.aput((client_id, "profile"), "current", sanitize_profile(profile))


def format_profile(profile: dict[str, str]) -> str:
    if not profile:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in profile.items())
