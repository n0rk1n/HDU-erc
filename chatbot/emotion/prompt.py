import json
import os
from pathlib import Path
from dataclasses import dataclass
from langchain_core.messages import SystemMessage
from chatbot.core.errors import ConfigError
from chatbot.core.paths import PROJECT_ROOT
from chatbot.emotion.config import read_json,content_hash
from chatbot.emotion.history import group_history
from chatbot.emotion.budget import select_history
from chatbot.emotion.retrieval import select_examples

DEFAULT_EMOTION_PROMPT_PATH = PROJECT_ROOT / "data" / "config" / "emotion_prompts.json"


def get_emotion_prompt() -> dict[str, str]:
    """Read and validate the prompt for each analysis; invalid config fails explicitly."""
    configured = os.getenv("EMOTION_SYSTEM_PROMPT_PATH", "").strip()
    path = Path(configured) if configured else DEFAULT_EMOTION_PROMPT_PATH
    data = read_json(path)
    if not isinstance(data, dict):
        raise ConfigError(f"emotion prompts config {path} must contain a JSON object")
    for key in ("version", "emotion_system"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ConfigError(f'emotion prompts config {path} requires a non-empty "{key}" string')
    return {key: data[key] for key in ("version", "emotion_system")}


@dataclass(frozen=True)
class PreparedAnalysis:
    messages: list
    snapshot: dict

def load_examples(path,taxonomy):
    data=read_json(path)
    if not isinstance(data,list) or not data:
        raise ConfigError('emotion examples must be a nonempty list')
    seen=set()
    for item in data:
        if not isinstance(item,dict) or any(not isinstance(item.get(k),str) or not item[k].strip() for k in ('id','dialogue','emotion')):
            raise ConfigError('invalid emotion example')
        if item['id'] in seen or item['emotion'] not in taxonomy.labels:
            raise ConfigError('duplicate example or unknown emotion')
        seen.add(item['id'])
    return data

def serialize_messages(messages, *, audit=False):
    roles={'human':'human' if audit else 'user','ai':'assistant','system':'system'}
    return [{'role':roles[m.type],'content':m.content,**({'id':m.id} if audit else {})} for m in messages]

def prepare_analysis(rows,*,current_id,taxonomy,examples,recent_labels,counter,budget):
    prompt_config = get_emotion_prompt()
    turns,current,excluded=group_history(rows,current_id=current_id)
    # Initial candidate retrieval uses the same bound; final pass accounts for actual instructions.
    candidate=select_history(turns,current,counter=counter,budget=budget,system_messages=[])
    likely=[label for label in recent_labels if label in taxonomy.labels]
    picked=select_examples(examples,json.dumps(serialize_messages(candidate.messages,audit=True),ensure_ascii=False),likely)
    system=SystemMessage(content=prompt_config['emotion_system']+'\n标签及描述：\n'+json.dumps(taxonomy.labels,ensure_ascii=False)+'\n参考示例：\n'+json.dumps(picked,ensure_ascii=False))
    candidate_ids=set(candidate.audit['retained_ids'])
    candidates=[turn for turn in turns if set(turn.message_ids)<=candidate_ids]
    selected=select_history(candidates,current,counter=counter,budget=budget,system_messages=[system])
    messages=[system,*selected.messages]
    removed=[m.id for t in turns for m in t.messages if m.id not in selected.audit['retained_ids']]
    return PreparedAnalysis(messages,{
        'prompt':serialize_messages(messages),'role_history':serialize_messages(selected.messages,audit=True),
        'prompt_version':prompt_config['version'],'prompt_config_hash':content_hash(prompt_config),'labels':taxonomy.labels,'families':taxonomy.families,'taxonomy_hash':taxonomy.content_hash,
        'examples':examples,'examples_hash':content_hash(examples),'selected_examples':picked,
        'retrieval':{'version':'weighted-overlap-v1','limit':4,'prior_boost':2.0,'recent_labels':likely,'candidate_ids':candidate.audit['retained_ids']},
        'budget':{**selected.audit,'removed_ids':removed,'excluded_incomplete_ids':excluded,'trim_reason':'token_budget' if removed else None},
    })
