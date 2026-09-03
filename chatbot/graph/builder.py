from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from functools import wraps
from typing import Any, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot.core.errors import InvalidMessageState
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
    compiled = build_turn_graph(deps).compile(checkpointer=checkpointer)
    return _bind_thread_validation(compiled)


def _bind_thread_validation(compiled: CompiledStateGraph) -> CompiledStateGraph:
    """Keep the official compiled type while rejecting mismatches before persistence."""
    original_ainvoke = compiled.ainvoke
    original_astream = compiled.astream

    @wraps(original_ainvoke)
    async def validated_ainvoke(
        input: object,
        config: object = None,
        *,
        context: object = None,
        **kwargs: object,
    ) -> object:
        _validate_thread_binding(config, context)
        return await original_ainvoke(
            input,
            cast(Any, config),
            context=cast(Any, context),
            **cast(dict[str, Any], kwargs),
        )

    @wraps(original_astream)
    def validated_astream(
        input: object,
        config: object = None,
        *,
        context: object = None,
        **kwargs: object,
    ) -> AsyncIterator[object]:
        _validate_thread_binding(config, context)
        return cast(
            AsyncIterator[object],
            original_astream(
                input,
                cast(Any, config),
                context=cast(Any, context),
                **cast(dict[str, Any], kwargs),
            ),
        )

    compiled.ainvoke = cast(Any, validated_ainvoke)
    compiled.astream = cast(Any, validated_astream)
    return compiled


def _validate_thread_binding(config: object, context: object) -> None:
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    configured_thread_id = (
        configurable.get("thread_id") if isinstance(configurable, Mapping) else None
    )
    if (
        not isinstance(context, TurnContext)
        or not isinstance(configured_thread_id, str)
        or configured_thread_id != context.thread_id
    ):
        raise InvalidMessageState(
            "configurable thread_id does not match TurnContext.thread_id"
        )
