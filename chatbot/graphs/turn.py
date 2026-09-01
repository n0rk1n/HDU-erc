"""Compiled conversation-turn workflow assembled from independently tested nodes."""

from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph

from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.nodes.context import load_context
from chatbot.graphs.nodes.emotion import (
    analyze_emotion_node,
    assess_safety_node,
    reuse_emotion,
)
from chatbot.graphs.nodes.finalization import finalize_turn, replay_completed_request
from chatbot.graphs.nodes.generation import generate_crisis_reply, generate_reply
from chatbot.graphs.nodes.input import accept_turn
from chatbot.graphs.nodes.memory import extract_memory, maybe_consolidate
from chatbot.graphs.nodes.risk import risk_precheck
from chatbot.graphs.routing import route_by_safety, route_emotion
from chatbot.graphs.state import ConversationState
from chatbot.models.graph import GraphContext


def build_turn_graph(deps: NodeDependencies) -> StateGraph:
    """Build the uncompiled main-turn graph around application-lifetime dependencies."""
    builder = StateGraph(ConversationState, context_schema=GraphContext)
    builder.add_node("accept_turn", accept_turn)
    builder.add_node("risk_precheck", risk_precheck)
    builder.add_node("analyze_emotion", partial(analyze_emotion_node, deps=deps))
    builder.add_node("reuse_emotion", reuse_emotion)
    builder.add_node("assess_safety", assess_safety_node)
    builder.add_node("load_context", partial(load_context, deps=deps))
    builder.add_node("generate_reply", partial(generate_reply, deps=deps))
    builder.add_node(
        "generate_crisis_reply",
        partial(generate_crisis_reply, deps=deps),
    )
    builder.add_node("extract_memory", partial(extract_memory, deps=deps))
    builder.add_node("maybe_consolidate", partial(maybe_consolidate, deps=deps))
    builder.add_node(
        "finalize_turn",
        partial(finalize_turn, graph_config=deps.graph_config),
    )
    builder.add_node("replay_completed_request", replay_completed_request)

    builder.add_edge(START, "accept_turn")
    builder.add_conditional_edges(
        "accept_turn",
        _route_after_accept,
        {
            "replay_completed_request": "replay_completed_request",
            "risk_precheck": "risk_precheck",
        },
    )
    builder.add_edge("replay_completed_request", END)
    builder.add_conditional_edges(
        "risk_precheck",
        partial(route_emotion, emotion_interval=deps.chat_config.emotion_interval),
        {
            "analyze_emotion": "analyze_emotion",
            "reuse_emotion": "reuse_emotion",
        },
    )
    builder.add_edge("analyze_emotion", "assess_safety")
    builder.add_edge("reuse_emotion", "assess_safety")
    builder.add_edge("assess_safety", "load_context")
    builder.add_conditional_edges(
        "load_context",
        route_by_safety,
        {
            "generate_reply": "generate_reply",
            "generate_crisis_reply": "generate_crisis_reply",
        },
    )
    builder.add_edge("generate_reply", "extract_memory")
    builder.add_edge("generate_crisis_reply", "extract_memory")
    builder.add_edge("extract_memory", "maybe_consolidate")
    builder.add_edge("maybe_consolidate", "finalize_turn")
    builder.add_edge("finalize_turn", END)
    return builder


def _route_after_accept(
    state: ConversationState,
) -> Literal["replay_completed_request", "risk_precheck"]:
    if state.get("replay_request") is True:
        return "replay_completed_request"
    return "risk_precheck"
