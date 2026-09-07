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
