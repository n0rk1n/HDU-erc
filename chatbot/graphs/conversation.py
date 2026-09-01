"""Operation-routing parent graph for all persistent conversation workflows."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import START, StateGraph

from chatbot.graphs.dependencies import NodeDependencies
from chatbot.graphs.nodes.input import GraphInputError
from chatbot.graphs.onboarding import build_onboarding_graph
from chatbot.graphs.regeneration import build_regeneration_graph
from chatbot.graphs.state import ConversationState
from chatbot.graphs.turn import build_turn_graph
from chatbot.models.graph import GraphContext


Route = Literal["turn", "regenerate", "onboard", "invalid"]


def build_conversation_graph(deps: NodeDependencies) -> StateGraph:
    """Build one lightweight parent around unpersisted compiled child graphs."""
    turn = build_turn_graph(deps).compile()
    regeneration = build_regeneration_graph(deps).compile()
    onboarding = build_onboarding_graph(deps).compile()

    builder = StateGraph(ConversationState, context_schema=GraphContext)
    builder.add_node("turn", turn)
    builder.add_node("regenerate", regeneration)
    builder.add_node("onboard", onboarding)
    builder.add_node("invalid", reject_invalid_operation)
    builder.add_conditional_edges(
        START,
        route_operation,
        {
            "turn": "turn",
            "regenerate": "regenerate",
            "onboard": "onboard",
            "invalid": "invalid",
        },
    )
    return builder


def route_operation(state: ConversationState) -> Route:
    """Route only the three public operations accepted by the graph contract."""
    operation = state.get("operation")
    if operation in {"turn", "regenerate", "onboard"}:
        return operation
    return "invalid"


def reject_invalid_operation(state: ConversationState) -> dict:
    """Raise the stable input-domain failure for unknown operations."""
    raise GraphInputError("invalid_operation")
