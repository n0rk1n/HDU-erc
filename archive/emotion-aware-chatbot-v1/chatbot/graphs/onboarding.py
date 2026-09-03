"""Draft a sanitized profile for user confirmation without persisting it."""

from __future__ import annotations

import json
from functools import partial
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.state import ConversationState
from chatbot.models.graph import GraphContext
from chatbot.profile.onboarding import (
    ALLOWED_PROFILE_FIELDS,
    _draft_prompt,
    fallback_profile_draft,
    sanitize_profile,
)


class ProfileDraft(TypedDict, total=False):
    preferred_name: str
    life_stage: str
    companion_expectation: str
    response_style: str
    avoidance: str


def build_onboarding_graph(deps: NodeDependencies) -> StateGraph:
    """Build the uncompiled profile-draft graph with no Store write node."""
    builder = StateGraph(ConversationState, context_schema=GraphContext)
    builder.add_node("validate_answers", validate_answers)
    builder.add_node("draft_profile", partial(draft_profile_node, deps=deps))
    builder.add_node("finalize_profile_draft", finalize_profile_draft)
    builder.add_edge(START, "validate_answers")
    builder.add_edge("validate_answers", "draft_profile")
    builder.add_edge("draft_profile", "finalize_profile_draft")
    builder.add_edge("finalize_profile_draft", END)
    return builder


def validate_answers(state: ConversationState) -> dict[str, Any]:
    """Discard malformed/unknown answers and normalize retained values."""
    raw_answers = state.get("profile_answers", [])
    answers = raw_answers if isinstance(raw_answers, list) else []
    mapping = fallback_profile_draft(
        [item for item in answers if isinstance(item, dict)]
    )
    cleaned = [
        {"key": key, "answer": mapping[key]}
        for key in ALLOWED_PROFILE_FIELDS
        if key in mapping
    ]
    return {"profile_answers": cleaned}


async def draft_profile_node(
    state: ConversationState,
    runtime,
    writer,
    config: RunnableConfig,
    *,
    deps: NodeDependencies,
) -> dict[str, dict[str, str]]:
    """Use structured async generation, degrading to the existing local fallback."""
    answers = list(state.get("profile_answers", []))
    fallback = fallback_profile_draft(answers)
    prompt = _draft_prompt(answers)
    try:
        model = deps.chat_model
        with_structured_output = getattr(model, "with_structured_output", None)
        if callable(with_structured_output):
            model = with_structured_output(ProfileDraft)
        result = await model.ainvoke(prompt, config=config)
        parsed = _profile_mapping(result)
        draft = sanitize_profile(parsed)
    except Exception:
        draft = fallback
    return {"profile_draft": draft or fallback}


def finalize_profile_draft(
    state: ConversationState,
    runtime,
    writer,
) -> dict[str, Any]:
    """Clear request-scoped fields while leaving profile_draft caller-visible."""
    return {
        "operation": "",
        "request_id": "",
        "input_message": "",
        "target_message_id": "",
        "regeneration_reason": "",
        "profile_answers": [],
        "risk": {},
        "profile_context": "",
        "memory_context": "",
        "safety_state": {},
        "response_content": "",
        "response_message_id": "",
        "error_code": "",
        "memory_warning": "",
        "replay_request": False,
    }


def _profile_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    content = result.content if hasattr(result, "content") else result
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("profile draft is not an object")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("profile draft is not an object")
    return parsed
