from dataclasses import dataclass
from collections import defaultdict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from chatbot.core.errors import InvalidMessageState

@dataclass(frozen=True)
class DialogueTurn:
    request_id: str
    messages: tuple[BaseMessage, BaseMessage]
    message_ids: tuple[str,str]

def group_history(rows, *, current_id):
    current=next((r for r in rows if r.id==current_id),None)
    if current is None or current.role!='user' or current.status!='completed':
        raise InvalidMessageState('invalid current emotion input')
    groups=defaultdict(list)
    for row in rows:
        if row.conversation_id != current.conversation_id:
            raise InvalidMessageState('cross-conversation emotion history')
        if row.sequence_no < current.sequence_no:
            groups[row.request_id].append(row)
    turns,excluded=[],[]
    for request,items in groups.items():
        items.sort(key=lambda r:r.sequence_no)
        if (len(items)==2 and [r.role for r in items]==['user','assistant']
            and all(r.status=='completed' for r in items)):
            a,b=items
            turns.append(DialogueTurn(request,(HumanMessage(content=a.content,id=a.id),AIMessage(content=b.content,id=b.id)),(a.id,b.id)))
        else:
            excluded.extend(r.id for r in items)
    return turns,HumanMessage(content=current.content,id=current.id),excluded
