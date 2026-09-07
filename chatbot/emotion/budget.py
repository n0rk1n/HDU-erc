from dataclasses import dataclass
from langchain_core.messages import BaseMessage

@dataclass(frozen=True)
class HistorySelection:
    messages: list[BaseMessage]
    audit: dict[str,object]

class ContextBudgetExceeded(ValueError):
    def __init__(self,audit):
        super().__init__('context_budget_exceeded')
        self.audit=audit

def select_history(turns,current,*,counter,budget,system_messages):
    cap=int(budget.context_tokens*budget.history_ratio)
    system_tokens=counter.count(system_messages) if system_messages else 0
    selected=[current]
    def fits(messages):
        return (counter.count(messages)<=cap and counter.count([*system_messages,*messages])+budget.output_tokens+budget.safety_tokens<=budget.context_tokens)
    def audit():
        ids=[m.id for m in selected]
        all_ids=[i for turn in turns for i in turn.message_ids]+[current.id]
        return {'context_tokens':budget.context_tokens,'history_ratio':budget.history_ratio,
                'history_token_cap':cap,'system_tokens':system_tokens,'output_reserve':budget.output_tokens,
                'safety_tokens':budget.safety_tokens,'effective_budget':min(cap,budget.context_tokens-system_tokens-budget.output_tokens-budget.safety_tokens),
                'counter':counter.identity,'counter_version':counter.version,
                'chat_tokens':counter.count(selected),'prompt_tokens':counter.count([*system_messages,*selected]),
                'retained_ids':ids,'removed_ids':[i for i in all_ids if i not in ids]}
    if not fits(selected):
        raise ContextBudgetExceeded(audit())
    for turn in reversed(turns):
        candidate=[*turn.messages,*selected]
        if not fits(candidate):
            break
        selected=candidate
    facts=audit()
    facts['trim_reason']='token_budget' if facts['removed_ids'] else None
    return HistorySelection(selected,facts)
