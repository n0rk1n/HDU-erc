from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from chatbot.core.errors import ConfigError
from chatbot.core.config import resolve_model_settings
from chatbot.core.paths import PROJECT_ROOT
from chatbot.core.prompt_config import load_prompt_config
from chatbot.emotion.config import EmotionSettings, read_json
from chatbot.emotion.types import BudgetConfig
from chatbot.emotion_gate.types import GatePolicy, GateSettings

DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'data/config/emotion_gate.json'
DEFAULT_PROMPT_PATH = PROJECT_ROOT / 'data/config/prompts/emotion_gate_prompts.json'


def load_gate_settings(emotion: EmotionSettings) -> GateSettings:
    def env(name, default=None):
        return os.getenv(name, '').strip() or default

    data = read_json(Path(env('EMOTION_GATE_CONFIG_PATH', str(DEFAULT_CONFIG_PATH))))
    policy_keys = {f.name for f in fields(GatePolicy)}
    budget_keys = {f.name for f in fields(BudgetConfig)}
    if (not isinstance(data, dict) or set(data) - policy_keys - budget_keys - {'version', 'tokenizer_model'}
            or not isinstance(data.get('version'), str) or not data['version'].strip()):
        raise ConfigError('invalid emotion gate config keys or version')
    try:
        policy = GatePolicy(**{k: data[k] for k in policy_keys if k in data})
        overrides = {k: data[k] for k in budget_keys if data.get(k) is not None}
        budget = BudgetConfig(**{k: overrides.get(k, getattr(emotion.budget, k)) for k in budget_keys})
        connection = resolve_model_settings('EMOTION_GATE_LLM')
        model = connection.model
        tokenizer = emotion.tokenizer_model if data.get('tokenizer_model') is None else data['tokenizer_model']
        if not isinstance(tokenizer, str) or not tokenizer.strip():
            raise ValueError('tokenizer_model must be a nonempty string')
        if model != emotion.model and (not data.get('tokenizer_model') or not data.get('context_tokens')):
            raise ValueError('different gate model requires explicit context_tokens and tokenizer_model')
    except (ValueError, TypeError) as exc:
        raise ConfigError(f'invalid emotion gate settings: {exc}') from exc
    path = Path(env('EMOTION_GATE_SYSTEM_PROMPT_PATH', str(DEFAULT_PROMPT_PATH)))
    return GateSettings(policy, connection.api_key, model, connection.base_url,
                        connection.temperature, connection.timeout_seconds,
                        budget, tokenizer, path, load_prompt_config(path), data['version'],
                        thinking=connection.thinking, reasoning_effort=connection.reasoning_effort)
