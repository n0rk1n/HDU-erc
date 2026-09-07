"""Explicit offline emotion dependencies; never installed in production."""
from chatbot.db.connection import Database
from chatbot.db.messages import MessageRepository
from chatbot.db.emotions import EmotionRepository
from chatbot.emotion.config import load_taxonomy,CONFIG_ROOT
from chatbot.emotion.prompt import load_examples
from chatbot.emotion.graph import EmotionRuntime
from chatbot.emotion.types import BudgetConfig,ModelOutcome
from chatbot.llm.types import TokenUsage

class Counter:
    identity='test-character-counter';version='1'
    def count(self,messages):return sum(len(m.content)+1 for m in messages)

class FailedEmotionModel:
    parameters={'model':'offline-emotion','max_retries':0}
    async def invoke(self,prompt):
        return ModelOutcome(None,None,TokenUsage(),None,{}, {'stage':'model','message':'offline failure'})

def emotion_runtime(database,model=None):
    taxonomy=load_taxonomy()
    return EmotionRuntime(EmotionRepository(database),MessageRepository(database),model or FailedEmotionModel(),Counter(),BudgetConfig(100000,1000,256),taxonomy,load_examples(CONFIG_ROOT/'emotion_examples.json',taxonomy))

def dependencies(**kwargs):
    from chatbot.graph import NodeDependencies
    return NodeDependencies(emotion=emotion_runtime(kwargs['messages'].database),**kwargs)

def create_app(config=None,model=None,**kwargs):
    from chatbot.web import create_app as real_create_app
    if config is not None and 'emotion' not in kwargs:
        kwargs['emotion']=emotion_runtime(Database(config.sqlite_db_path))
    return real_create_app(config=config,model=model,**kwargs)
