import json
from chatbot.emotion.types import ModelOutcome
from chatbot.llm.types import TokenUsage

class GateModel:
    parameters={'model':'offline-gate','max_retries':0}
    def __init__(self, outputs=None):
        self.outputs=iter(outputs or ['{"should_analyze":false,"reason":"stable"}'] * 50)
        self.prompts=[]
    async def invoke(self,messages):
        self.prompts.append(messages)
        output=next(self.outputs)
        if isinstance(output,BaseException):raise output
        return ModelOutcome(output,None,TokenUsage(3,2,5),'stop',{},None)

async def complete(messages,turn):
    await messages.mark_streaming(turn.assistant.id)
    await messages.complete_assistant(turn.assistant.id,content='reply',reasoning_content=None,
        trace={},prompt=[],provider=None,model=None,parameters=None,input_tokens=None,
        output_tokens=None,total_tokens=None,latency_ms=None,finish_reason=None)


def gate_runtime(database,model=None,policy=None):
    from chatbot.db.emotion_gates import GateRepository
    from chatbot.db.messages import MessageRepository
    from chatbot.db.emotions import EmotionRepository
    from chatbot.emotion_gate.types import GatePolicy,GateSettings
    from chatbot.emotion_gate.runtime import GateRuntime
    from chatbot.emotion_gate.config import DEFAULT_PROMPT_PATH
    from chatbot.core.prompt_config import load_prompt_config
    from chatbot.emotion.types import BudgetConfig
    from tests.emotion.helpers import Counter
    from pydantic import SecretStr
    settings=GateSettings(policy or GatePolicy(),SecretStr('offline'),'offline-gate',None,0,1,
        BudgetConfig(100000,1000,256),'offline',DEFAULT_PROMPT_PATH,load_prompt_config(DEFAULT_PROMPT_PATH))
    return GateRuntime(GateRepository(database),MessageRepository(database),EmotionRepository(database),model or GateModel(),Counter(),settings)
