from __future__ import annotations
import hashlib
import json
from pathlib import Path
from chatbot.core.errors import ConfigError
from chatbot.emotion.types import Taxonomy

CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'config'

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f'duplicate key: {key}')
        result[key] = value
    return result

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
    except (OSError, ValueError) as exc:
        raise ConfigError(f'cannot read emotion config: {path.name}') from exc

def content_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()

def load_taxonomy(labels_path=CONFIG_ROOT/'emotion_labels.json', families_path=CONFIG_ROOT/'emotion_families.json'):
    labels, families = read_json(labels_path), read_json(families_path)
    for mapping in (labels, families):
        if not isinstance(mapping, dict) or not mapping:
            raise ConfigError('emotion config must be a nonempty object')
        if any(not k.strip() or not isinstance(v, str) or not v.strip() for k,v in mapping.items()):
            raise ConfigError('emotion labels and values must be nonempty strings')
    if labels.keys() != families.keys():
        raise ConfigError('emotion family keys must exactly match labels')
    return Taxonomy(labels, families, content_hash({'labels':labels, 'families':families}))

from dataclasses import dataclass
import os
from pydantic import SecretStr
from chatbot.emotion.types import BudgetConfig

@dataclass(frozen=True)
class EmotionSettings:
    api_key: SecretStr
    model: str
    base_url: str | None
    temperature: float
    timeout_seconds: float
    budget: BudgetConfig
    tokenizer_model: str
    labels_path: Path
    families_path: Path
    examples_path: Path

def load_emotion_settings(chat_config):
    def env(name, default=None):
        return os.getenv(name, '').strip() or default
    context=env('EMOTION_CONTEXT_TOKENS')
    tokenizer=env('EMOTION_TOKENIZER_MODEL')
    if not context:
        raise ConfigError('EMOTION_CONTEXT_TOKENS is required for the deployed emotion model')
    if not tokenizer:
        raise ConfigError('EMOTION_TOKENIZER_MODEL is required and must match the deployed model')
    try:
        budget=BudgetConfig(int(context),int(env('EMOTION_OUTPUT_TOKENS','1024')),int(env('EMOTION_SAFETY_TOKENS','256')),float(env('EMOTION_HISTORY_RATIO','0.60')))
        temperature=float(env('EMOTION_LLM_TEMPERATURE','0'))
        timeout=float(env('EMOTION_LLM_TIMEOUT_SECONDS',str(chat_config.llm_timeout_seconds)))
        if not 0<=temperature<=2 or not 0<timeout<float('inf'):
            raise ValueError()
    except ValueError as exc:
        raise ConfigError('invalid emotion model parameters') from exc
    return EmotionSettings(SecretStr(env('EMOTION_LLM_API_KEY',chat_config.llm_api_key.get_secret_value())),
        env('EMOTION_LLM_MODEL',chat_config.llm_model),env('EMOTION_LLM_BASE_URL',chat_config.llm_base_url),temperature,timeout,budget,tokenizer,
        Path(env('EMOTION_LABELS_PATH',str(CONFIG_ROOT/'emotion_labels.json'))),
        Path(env('EMOTION_FAMILIES_PATH',str(CONFIG_ROOT/'emotion_families.json'))),
        Path(env('EMOTION_EXAMPLES_PATH',str(CONFIG_ROOT/'emotion_examples.json'))))
