from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from pydantic import SecretStr
from chatbot.emotion.types import BudgetConfig


@dataclass(frozen=True)
class GatePolicy:
    max_interval_turns: int = 15
    history_turn_limit: int = 5
    force_first_analysis: bool = True
    gate_failure_action: str = 'analyze'
    retry_failed_analysis_next_turn: bool = True
    reuse_last_success_on_skip: bool = True
    reuse_last_success_on_analysis_failure: bool = False
    model_retry_count: int = 0
    retry_delay_seconds: float = 0

    def __post_init__(self):
        for name, minimum in [('max_interval_turns', 1), ('history_turn_limit', 0), ('model_retry_count', 0)]:
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name in ('force_first_analysis', 'retry_failed_analysis_next_turn',
                     'reuse_last_success_on_skip', 'reuse_last_success_on_analysis_failure'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.gate_failure_action not in ('analyze', 'skip'):
            raise ValueError('gate_failure_action must be analyze or skip')
        if (type(self.retry_delay_seconds) not in (int, float)
                or not isfinite(self.retry_delay_seconds) or self.retry_delay_seconds < 0):
            raise ValueError('retry_delay_seconds must be a finite nonnegative number')


@dataclass(frozen=True)
class GateSettings:
    policy: GatePolicy
    api_key: SecretStr
    model: str
    base_url: str | None
    temperature: float
    timeout_seconds: float
    budget: BudgetConfig
    tokenizer_model: str
    prompts_path: Path
    prompt: dict[str, str]
    version: str = 'v1'
    thinking: str = 'disabled'

    @property
    def wait_seconds(self) -> float:
        return ((1 + self.policy.model_retry_count) * self.timeout_seconds
                + self.policy.model_retry_count * self.policy.retry_delay_seconds)


@dataclass(frozen=True)
class GateResult:
    should_analyze: bool
    reason: str
