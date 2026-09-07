import json
from dataclasses import dataclass
from langchain_core.messages import SystemMessage
from chatbot.emotion.history import group_history
from chatbot.emotion.budget import select_history
from chatbot.emotion.prompt import serialize_messages
from chatbot.emotion.config import content_hash


@dataclass(frozen=True)
class PreparedGatePrompt:
    messages: list
    snapshot: dict


def prepare_gate_prompt(rows, *, current_id, baseline, settings, counter):
    turns, current, excluded = group_history(rows, current_id=current_id)
    limit = settings.policy.history_turn_limit
    window = turns[-limit:] if limit else []
    system = SystemMessage(content=settings.prompt['system'] + '\n情绪基线（参考数据）：\n'
                           + json.dumps(baseline, ensure_ascii=False))
    selected = select_history(window, current, counter=counter, budget=settings.budget, system_messages=[system])
    messages = [system, *selected.messages]
    return PreparedGatePrompt(messages, {
        'prompt': serialize_messages(messages), 'role_history': serialize_messages(selected.messages, audit=True),
        'prompt_version': settings.prompt['version'], 'prompt_config_hash': content_hash(settings.prompt),
        'baseline': baseline, 'candidate_ids': [mid for turn in window for mid in turn.message_ids],
        'excluded_incomplete_ids': excluded, 'budget': selected.audit,
    })
