from __future__ import annotations

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot.graph.dependencies import NodeDependencies
from chatbot.graph.nodes import TurnNodes
from chatbot.graph.state import TurnContext, TurnState


def build_turn_graph(deps: NodeDependencies) -> StateGraph:
    nodes = TurnNodes(deps)
    builder = StateGraph(TurnState, context_schema=TurnContext)
    builder.add_node("prepare_turn", nodes.prepare_turn)
    builder.add_node("generate_response", nodes.generate_response)
    builder.add_node("finalize_turn", nodes.finalize_turn)
    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "generate_response")
    builder.add_edge("generate_response", "finalize_turn")
    builder.add_edge("finalize_turn", END)
    return builder


def compile_turn_graph(
    deps: NodeDependencies, checkpointer: AsyncSqliteSaver
) -> CompiledStateGraph:
    return build_turn_graph(deps).compile(checkpointer=checkpointer)
