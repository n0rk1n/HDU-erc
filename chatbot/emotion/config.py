from __future__ import annotations
import hashlib
import json
from pathlib import Path
from chatbot.core.errors import ConfigError
from chatbot.core.config import resolve_model_settings
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

def emotion_labels_path() -> Path:
    return Path(os.getenv('EMOTION_LABELS_PATH', '').strip() or CONFIG_ROOT / 'emotion_labels.json')


def load_label_definitions(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    entries = read_json(path)
    if not isinstance(entries, dict) or not entries:
        raise ConfigError('emotion labels must be a nonempty object')
    labels, names = {}, {}
    for label, entry in entries.items():
        if not label.strip():
            raise ConfigError('emotion labels must be nonempty strings')
        if isinstance(entry, str):
            description = entry
        elif isinstance(entry, dict) and not entry.keys() - {'description', 'display_name'}:
            description = entry.get('description')
            if 'display_name' in entry:
                name = entry['display_name']
                if not isinstance(name, str) or not name.strip():
                    raise ConfigError('emotion display_name must be a nonempty string')
                names[label] = name
        else:
            raise ConfigError('emotion label must be a description string or description/display_name object')
        if not isinstance(description, str) or not description.strip():
            raise ConfigError('emotion description must be a nonempty string')
        labels[label] = description
    return labels, names


def load_taxonomy(labels_path=CONFIG_ROOT/'emotion_labels.json', families_path=CONFIG_ROOT/'emotion_families.json'):
    labels, _ = load_label_definitions(labels_path)
    families = read_json(families_path)
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
from chatbot.emotion.types import BudgetConfig, RetrievalConfig

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
    retrieval: RetrievalConfig = RetrievalConfig()

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
        retrieval=RetrievalConfig(int(env('EMOTION_EXAMPLE_LIMIT','4')),float(env('EMOTION_PRIOR_BOOST','2.0')),int(env('EMOTION_RECENT_LABEL_LIMIT','3')))
    except ValueError as exc:
        raise ConfigError('invalid emotion model parameters') from exc
    connection = resolve_model_settings('EMOTION_LLM')
    return EmotionSettings(connection.api_key, connection.model, connection.base_url,
        connection.temperature, connection.timeout_seconds, budget, tokenizer,
        emotion_labels_path(),
        Path(env('EMOTION_FAMILIES_PATH',str(CONFIG_ROOT/'emotion_families.json'))),
        Path(env('EMOTION_EXAMPLES_PATH',str(CONFIG_ROOT/'emotion_examples.json'))),retrieval)
