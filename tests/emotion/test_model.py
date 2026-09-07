from types import SimpleNamespace
import pytest
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, AIMessage
from chatbot.emotion.types import BudgetConfig

@pytest.fixture
def settings():
    return SimpleNamespace(api_key=SecretStr('secret-test-key'),model='gpt-4o-mini',base_url=None,temperature=0,timeout_seconds=1,budget=BudgetConfig(4096,512,64),tokenizer_model='gpt-4o-mini')

async def test_provider_failure_keeps_safe_details(settings):
    from chatbot.emotion.model import OpenAICompatibleEmotionModel
    class Failure(Exception):
        status_code=429
        request_id='provider-r1'
        body={'error':{'message':'secret-test-key','code':'rate_limit'},'authorization':'bearer secret'}
    class Client:
        async def ainvoke(self,messages):
            raise Failure('failed secret-test-key')
    model=OpenAICompatibleEmotionModel(settings,client=Client())
    outcome=await model.invoke([HumanMessage(content='hi')])
    assert outcome.error['http_status']==429
    assert outcome.error['provider_request_id']=='provider-r1'
    assert 'secret-test-key' not in str(outcome)
    assert 'bearer secret' not in str(outcome)
    assert outcome.usage.input_tokens is None
    assert model.parameters['max_retries']==0

async def test_success_preserves_provider_output(settings):
    from chatbot.emotion.model import OpenAICompatibleEmotionModel
    class Client:
        async def ainvoke(self,messages):
            return AIMessage(content='{}',additional_kwargs={'reasoning_content':'provided reasoning'},usage_metadata={'input_tokens':4,'output_tokens':2,'total_tokens':6})
    outcome=await OpenAICompatibleEmotionModel(settings,client=Client()).invoke([])
    assert outcome.raw_output=='{}'
    assert outcome.reasoning_content=='provided reasoning'
    assert outcome.usage.total_tokens==6


def test_missing_context_configuration_rejected(monkeypatch):
    from chatbot.emotion.config import load_emotion_settings
    from chatbot.core.errors import ConfigError
    monkeypatch.delenv('EMOTION_CONTEXT_TOKENS',raising=False)
    with pytest.raises(ConfigError,match='EMOTION_CONTEXT_TOKENS'):
        load_emotion_settings(SimpleNamespace())
