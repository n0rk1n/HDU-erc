from __future__ import annotations

import os
from dataclasses import fields
from math import isfinite
from pathlib import Path
from pydantic import SecretStr
from chatbot.core.errors import ConfigError
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
        model = env('EMOTION_GATE_LLM_MODEL', emotion.model)
        tokenizer = emotion.tokenizer_model if data.get('tokenizer_model') is None else data['tokenizer_model']
        if not isinstance(tokenizer, str) or not tokenizer.strip():
            raise ValueError('tokenizer_model must be a nonempty string')
        if model != emotion.model and (not data.get('tokenizer_model') or not data.get('context_tokens')):
            raise ValueError('different gate model requires explicit context_tokens and tokenizer_model')
        temperature = float(env('EMOTION_GATE_LLM_TEMPERATURE', str(emotion.temperature)))
        timeout = float(env('EMOTION_GATE_LLM_TIMEOUT_SECONDS', str(emotion.timeout_seconds)))
        if not isfinite(temperature) or not 0 <= temperature <= 2 or not isfinite(timeout) or timeout <= 0:
            raise ValueError('invalid gate temperature or timeout')
    except (ValueError, TypeError) as exc:
        raise ConfigError(f'invalid emotion gate settings: {exc}') from exc
    path = Path(env('EMOTION_GATE_SYSTEM_PROMPT_PATH', str(DEFAULT_PROMPT_PATH)))
    return GateSettings(policy, SecretStr(env('EMOTION_GATE_LLM_API_KEY', emotion.api_key.get_secret_value())),
                        model, env('EMOTION_GATE_LLM_BASE_URL', emotion.base_url), temperature, timeout,
                        budget, tokenizer, path, load_prompt_config(path), data['version'])
