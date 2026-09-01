"""Client-scoped profile and long-term-memory context loading."""

import logging
from typing import Any

from chatbot.emotion.state import EmotionState
from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.state import ConversationState
from chatbot.memory import DEFAULT_MEMORY_MAX_RESULTS, format_memory_context
from chatbot.memory.consolidation import build_memory_search_query
from chatbot.profile import format_profile, load_profile


logger = logging.getLogger(__name__)


async def load_context(
    state: ConversationState,
    runtime,
    writer,
    *,
    deps: NodeDependencies,
) -> dict[str, str]:
    """Load plain-text context for exactly the runtime client, degrading read failures."""
    client_id = runtime.context["client_id"]

    try:
        profile = await load_profile(runtime.store, client_id)
        profile_context = format_profile(profile)
    except Exception:
        logger.warning("profile context read failed", exc_info=True)
        profile_context = ""

    query = build_memory_search_query(
        state.get("input_message", ""),
        _emotion_from_state(state.get("emotion_state")),
        state.get("recent_emotions", []),
    )
    try:
        memories = await deps.memory_repository.asearch(
            client_id,
            query,
            limit=DEFAULT_MEMORY_MAX_RESULTS,
        )
        memory_context = format_memory_context(memories)
    except Exception:
        logger.warning("memory context read failed", exc_info=True)
        memory_context = ""

    return {
        "profile_context": profile_context,
        "memory_context": memory_context,
    }


def _emotion_from_state(value: dict[str, Any] | None) -> EmotionState | None:
    if not isinstance(value, dict):
        return None
    return EmotionState.from_mapping(value)
