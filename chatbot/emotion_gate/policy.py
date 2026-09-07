from chatbot.emotion_gate.types import GatePolicy


def forced_reason(policy: GatePolicy, *, current_turn: int,
                  last_success_turn: int | None, latest_analysis_failed: bool) -> str | None:
    if current_turn == 1 and policy.force_first_analysis:
        return 'first_turn'
    if latest_analysis_failed and policy.retry_failed_analysis_next_turn:
        return 'previous_analysis_failed'
    elapsed = current_turn if last_success_turn is None else current_turn - last_success_turn
    if elapsed >= policy.max_interval_turns:
        return 'interval_reached'
    return None
