"""Non-fatal long-term-memory side-effect nodes."""

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.state import ConversationState
from chatbot.memory import load_memory_config
from chatbot.memory.consolidation import (
    consolidation_due,
    extract_consolidated_memory_candidates,
    load_memory_consolidation_config,
    recent_consolidation_window,
)
from chatbot.memory.extractor import extract_memory_candidates


logger = logging.getLogger(__name__)


async def extract_memory(
    state: ConversationState,
    runtime,
    writer,
    config: RunnableConfig,
    *,
    deps: NodeDependencies,
) -> dict[str, str]:
    """Extract and idempotently persist durable candidates from a completed turn."""
    candidates = extract_memory_candidates(
        state.get("input_message", ""),
        state.get("response_content", ""),
    )
    if not candidates:
        return {"memory_warning": ""}
    try:
        await deps.memory_repository.aremember(runtime.context["client_id"], candidates)
    except Exception:
        _log_failure("memory extraction write failed", state, runtime, config)
        return {"memory_warning": "memory_write_failed"}
    return {"memory_warning": ""}


async def maybe_consolidate(
    state: ConversationState,
    runtime,
    writer,
    config: RunnableConfig,
    *,
    deps: NodeDependencies,
) -> dict[str, str]:
    """Consolidate a due message window once per source checkpoint."""
    client_id = runtime.context["client_id"]
    request_id = state.get("request_id") or runtime.context["request_id"]
    source_checkpoint_id = _source_checkpoint_id(config, request_id)
    consolidation_config = load_memory_consolidation_config(load_memory_config())
    try:
        consolidation_state = await deps.memory_repository.aget_consolidation_state(
            client_id
        )
        if source_checkpoint_id in consolidation_state["processed_checkpoint_ids"]:
            return {"memory_warning": ""}
        turn_count = state.get("turn_count", 0)
        if not consolidation_due(
            consolidation_config,
            turn_count=turn_count,
            last_turn_count=consolidation_state["last_turn_count"],
        ):
            return {"memory_warning": ""}
        records = recent_consolidation_window(
            _message_records(state.get("messages", [])),
            window=consolidation_config.window,
            last_message_id=consolidation_state["last_message_id"],
        )
        candidates = extract_consolidated_memory_candidates(records)
        if candidates:
            await deps.memory_repository.aremember(client_id, candidates)
        await deps.memory_repository.amark_consolidated(
            client_id,
            turn_count=turn_count,
            last_message_id=_last_message_id(records),
            source_checkpoint_id=source_checkpoint_id,
        )
    except Exception:
        _log_failure("memory consolidation failed", state, runtime, config)
        return {"memory_warning": "memory_consolidation_failed"}
    return {"memory_warning": ""}


def _source_checkpoint_id(config: RunnableConfig, request_id: str) -> str:
    checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
    if isinstance(checkpoint_id, str) and checkpoint_id:
        return checkpoint_id
    return f"request:{request_id}"


def _message_records(messages: list[BaseMessage]) -> list[dict[str, str]]:
    records = []
    for message in messages:
        content_value = message.content
        content = (
            content_value.strip()
            if isinstance(content_value, str)
            else str(content_value).strip()
        )
        records.append(
            {
                "id": str(message.id or ""),
                "role": message.type,
                "content": content,
            }
        )
    return records


def _last_message_id(records: list[dict[str, str]]) -> str | None:
    for record in reversed(records):
        if record["id"]:
            return record["id"]
    return None


def _log_failure(
    message: str,
    state: ConversationState,
    runtime,
    config: RunnableConfig,
) -> None:
    configurable: dict[str, Any] = config.get("configurable", {})
    thread_id = state.get("thread_meta", {}).get("thread_id") or configurable.get(
        "thread_id", ""
    )
    logger.warning(
        "%s request_id=%s thread_id=%s",
        message,
        state.get("request_id", runtime.context.get("request_id", "")),
        thread_id,
        exc_info=True,
    )
