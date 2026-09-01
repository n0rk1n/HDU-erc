"""Deterministic, browser-safe risk precheck node."""

from chatbot.emotion import precheck_risk
from chatbot.graphs.state import ConversationState


def risk_precheck(state: ConversationState, runtime, writer) -> dict:
    """Calculate risk codes without emitting matched source phrases."""
    return {"risk": precheck_risk(state.get("input_message", ""))}
