"""Scheduling and model failures are separate from storage failures."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from time import monotonic
from chatbot.core.errors import InvalidMessageState
from chatbot.db.emotion_gates import GateRepository, GateDecision
from chatbot.db.emotions import EmotionRepository
from chatbot.db.messages import MessageRepository
from chatbot.emotion.types import EmotionModel, TokenCounter
from chatbot.emotion.model import safe_facts, error_facts
from chatbot.emotion_gate.types import GateSettings
from chatbot.emotion_gate.policy import forced_reason
from chatbot.emotion_gate.prompt import prepare_gate_prompt
from chatbot.emotion_gate.parsing import parse_gate_result


@dataclass(frozen=True)
class GateRuntime:
    repository: GateRepository
    messages: MessageRepository
    emotions: EmotionRepository
    model: EmotionModel
    counter: TokenCounter
    settings: GateSettings


async def decide_emotion(runtime: GateRuntime, *, conversation_id: str, request_id: str,
                         user_message_id: str) -> GateDecision:
    repo = runtime.repository
    row, created = await repo.reserve(conversation_id, request_id, user_message_id)
    if not created:
        if row.status != 'completed' or row.action is None:
            raise InvalidMessageState('gate in progress or interrupted; recover request first')
        return row
    try:
        return await _decide(runtime, row)
    except asyncio.CancelledError:
        try:
            await repo.interrupt(row.id)
        except Exception:
            logging.getLogger(__name__).warning('Unable to persist cancelled gate %s', row.id)
        raise


async def _decide(runtime, row):
    repo, settings = runtime.repository, runtime.settings
    policy = settings.policy
    secret = getattr(runtime.model, 'secret', '')
    facts = await repo.facts(row.conversation_id, row.user_message_id)
    baseline = None
    if facts.last_success_analysis_id:
        analysis = await runtime.emotions.get(facts.last_success_analysis_id)
        baseline = {'analysis_id': analysis.id, 'user_message_id': analysis.user_message_id,
                    'source_turn': facts.last_success_turn, 'elapsed_turns': facts.current_turn - facts.last_success_turn,
                    'result': analysis.result_json}
    snapshot = {'facts': asdict(facts), 'policy': asdict(policy), 'config_version': settings.version,
                'model_parameters': runtime.model.parameters, 'budget_config': asdict(settings.budget),
                'prompt_config': settings.prompt, 'baseline': baseline}
    await repo.start(row.id, snapshot=safe_facts(snapshot, secret))
    reason = forced_reason(policy, current_turn=facts.current_turn,
                           last_success_turn=facts.last_success_turn, latest_analysis_failed=facts.latest_analysis_failed)
    if reason:
        await repo.finish(row.id, action='analyze', reason=reason)
        return await repo.get(row.id)

    # Queries deliberately outside the model/format failure boundary.
    turn = await runtime.messages.find_turn(row.conversation_id, row.request_id)
    rows = await runtime.messages.list_emotion_history(row.conversation_id, through_sequence=turn.user.sequence_no)
    try:
        prepared = prepare_gate_prompt(rows, current_id=row.user_message_id, baseline=baseline,
                                       settings=settings, counter=runtime.counter)
    except Exception as exc:
        error = error_facts(exc, stage='prepare', secret=secret)
        if hasattr(exc, 'audit'):
            error['budget'] = exc.audit
        await repo.finish(row.id, action=policy.gate_failure_action, reason='gate_failed', error=error)
        return await repo.get(row.id)

    for number in range(policy.model_retry_count + 1):
        if number:
            await asyncio.sleep(policy.retry_delay_seconds)
        attempt = await repo.start_attempt(row.id, snapshot=safe_facts({**snapshot, **prepared.snapshot}, secret))
        started = monotonic()
        error, result = None, None
        attempt_facts = {}
        try:
            outcome = await runtime.model.invoke(prepared.messages)
        except Exception as exc:
            error = error_facts(exc, stage='model', secret=secret)
        else:
            attempt_facts.update(asdict(outcome))
            error = outcome.error
            if not error:
                try:
                    result = parse_gate_result(outcome.raw_output or '')
                except (ValueError, TypeError) as exc:
                    error = error_facts(exc, stage='parse', secret=secret)
        attempt_facts.update(latency_ms=int((monotonic() - started) * 1000), error=error,
                             result=asdict(result) if result else None)
        await repo.finish_attempt(attempt, status='failed' if error else 'completed', facts=safe_facts(attempt_facts, secret))
        if result:
            await repo.finish(row.id, action='analyze' if result.should_analyze else 'skip',
                              reason='emotion_changed' if result.should_analyze else 'unchanged')
            return await repo.get(row.id)
    await repo.finish(row.id, action=policy.gate_failure_action, reason='gate_failed', error=safe_facts(error, secret))
    return await repo.get(row.id)
