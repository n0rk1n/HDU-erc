"""Single non-retrying model call with safe, structured diagnostics."""
from __future__ import annotations
import re
from importlib.metadata import version
from langchain_openai import ChatOpenAI
from chatbot.core.errors import ConfigError
from chatbot.emotion.types import ModelOutcome
from chatbot.llm.types import TokenUsage
from chatbot.llm.openai_compatible import _text_content, _reasoning_content, _token_usage
from chatbot.llm.redaction import redact_secrets


def safe_facts(value, secret=''):
    value=redact_secrets(value)
    if isinstance(value,str):
        if secret:
            value=value.replace(secret,'[REDACTED]')
        value=re.sub(r'(?i)(bearer\s+)\S+',r'\1[REDACTED]',value)
        value=re.sub(r'https?://[^\s<>"\']+',lambda m:str(redact_secrets({'url':m[0]})['url']),value)
        return value
    if isinstance(value,dict):
        return {k:safe_facts(v,secret) for k,v in value.items()}
    if isinstance(value,list):
        return [safe_facts(v,secret) for v in value]
    return value


def error_facts(exc,*,stage,secret=''):
    response=getattr(exc,'response',None)
    headers=getattr(response,'headers',{}) or {}
    body=getattr(exc,'body',None)
    if body is None and response is not None:
        try:
            body=response.json()
        except Exception:
            body=getattr(response,'text',None)
    return safe_facts({'stage':stage,'type':type(exc).__name__,'message':str(exc),
        'code':getattr(exc,'code',None),
        'http_status':getattr(exc,'status_code',None) or getattr(response,'status_code',None),
        'provider_request_id':getattr(exc,'request_id',None) or headers.get('x-request-id'),
        'body':body},secret)

class OpenAICompatibleEmotionModel:
    def __init__(self,settings,*,client=None):
        self.secret=settings.api_key.get_secret_value()
        self.parameters=safe_facts({'provider':'openai-compatible','model':settings.model,'base_url':settings.base_url,
            'temperature':settings.temperature,'timeout':settings.timeout_seconds,'max_tokens':settings.budget.output_tokens,
            'max_retries':0,'streaming':False,'tokenizer_model':settings.tokenizer_model},self.secret)
        self.client=client if client is not None else ChatOpenAI(api_key=settings.api_key,model=settings.model,
            base_url=settings.base_url,temperature=settings.temperature,timeout=settings.timeout_seconds,
            max_tokens=settings.budget.output_tokens,max_retries=0,streaming=False,tiktoken_model_name=settings.tokenizer_model)

    async def invoke(self,messages):
        try:
            reply=await self.client.ainvoke(messages)
        except Exception as exc:
            error=error_facts(exc,stage='model',secret=self.secret)
            body=error.get('body')
            usage=_token_usage(None,body) if isinstance(body,dict) else None
            return ModelOutcome(None,None,usage or TokenUsage(),None,{},error)
        return ModelOutcome(safe_facts(_text_content(reply.content),self.secret),
            safe_facts(_reasoning_content(reply.additional_kwargs),self.secret) or None,
            _token_usage(reply.usage_metadata,reply.response_metadata) or TokenUsage(),
            reply.response_metadata.get('finish_reason'),
            safe_facts({'response_id':reply.id,**reply.response_metadata},self.secret),None)

class ModelTokenCounter:
    def __init__(self,client,*,tokenizer_model):
        # LangChain can silently fall back to another encoding: explicitly reject that.
        import tiktoken
        try:
            tiktoken.encoding_for_model(tokenizer_model)
            from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
            client.get_num_tokens_from_messages([SystemMessage(content='s'),HumanMessage(content='中文'),AIMessage(content='a')])
        except Exception as exc:
            raise ConfigError('unsupported emotion tokenizer; configure a matching model counter') from exc
        self.client=client
        self.identity=tokenizer_model
        self.version='langchain-openai:'+version('langchain-openai')+';tiktoken:'+version('tiktoken')
    def count(self,messages):
        return self.client.get_num_tokens_from_messages(list(messages)) if messages else 0
