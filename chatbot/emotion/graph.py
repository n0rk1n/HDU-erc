"""Uncheckpointed analysis subgraph; SQLite owns durable facts."""
from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass,asdict
from time import monotonic
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from chatbot.db.emotions import EmotionRepository
from chatbot.db.messages import MessageRepository
from chatbot.emotion.types import EmotionModel,TokenCounter,BudgetConfig,Taxonomy,ModelOutcome,RetrievalConfig
from chatbot.emotion.prompt import prepare_analysis
from chatbot.emotion.parsing import parse_result,ResultValidationError
from chatbot.emotion.model import error_facts,safe_facts
from chatbot.emotion.budget import ContextBudgetExceeded
from chatbot.core.errors import InvalidMessageState

@dataclass(frozen=True)
class EmotionRuntime:
    repository: EmotionRepository
    messages: MessageRepository
    model: EmotionModel
    counter: TokenCounter
    budget: BudgetConfig
    taxonomy: Taxonomy
    examples: list[dict[str,str]]
    retrieval: RetrievalConfig = RetrievalConfig()

class AnalysisState(TypedDict,total=False):
    conversation_id: str
    request_id: str
    user_message_id: str
    analysis_id: str
    analysis_status: str
    prepared: object
    snapshot: dict
    outcome: ModelOutcome
    result: dict
    error: dict
    started: float
    replay: bool

def build_emotion_graph(runtime: EmotionRuntime):
    repo=runtime.repository
    secret=getattr(runtime.model,'secret','')

    async def prepare(state):
        row,created=await repo.reserve(state['conversation_id'],state['request_id'],state['user_message_id'])
        base={'analysis_id':row.id,'analysis_status':row.status,'started':monotonic()}
        if not created:
            if row.status not in ('completed','failed'):
                raise InvalidMessageState('emotion analysis already in progress; recover interrupted run first')
            return {**base,'replay':True}
        snapshot={'labels':runtime.taxonomy.labels,'families':runtime.taxonomy.families,
            'taxonomy_hash':runtime.taxonomy.content_hash,'examples':runtime.examples,
            'model_parameters':runtime.model.parameters,'budget_config':asdict(runtime.budget),
            'retrieval_config':asdict(runtime.retrieval),
            'counter':runtime.counter.identity,'counter_version':runtime.counter.version}
        # DB errors intentionally propagate; they are not ordinary model failures.
        turn=await runtime.messages.find_turn(state['conversation_id'],state['request_id'])
        rows=await runtime.messages.list_emotion_history(state['conversation_id'],through_sequence=turn.user.sequence_no)
        recent=await repo.recent_labels(state['conversation_id'],before_sequence=turn.user.sequence_no,limit=runtime.retrieval.recent_label_limit)
        snapshot['source_message_ids']=[r.id for r in rows]
        try:
            prepared=prepare_analysis(rows,current_id=state['user_message_id'],taxonomy=runtime.taxonomy,
                examples=runtime.examples,recent_labels=recent,counter=runtime.counter,budget=runtime.budget,retrieval=runtime.retrieval)
            snapshot.update(prepared.snapshot)
        except Exception as exc:
            if isinstance(exc,ContextBudgetExceeded):
                snapshot['budget']=exc.audit
            error=error_facts(exc,stage='prepare',secret=secret)
            if isinstance(exc,ContextBudgetExceeded): error['code']='context_budget_exceeded'
            await repo.finish(row.id,status='failed',facts={'snapshot':safe_facts(snapshot,secret),'error':error,'latency_ms':int((monotonic()-base['started'])*1000)})
            return {**base,'replay':True,'analysis_status':'failed'}
        snapshot=safe_facts(snapshot,secret)
        await repo.start(row.id,snapshot=snapshot)
        return {**base,'prepared':prepared,'snapshot':snapshot,'analysis_status':'running','replay':False}

    async def call(state):
        if state.get('replay'): return {}
        try:
            outcome=await runtime.model.invoke(state['prepared'].messages)
        except asyncio.CancelledError:
            try:
                await repo.finish(state['analysis_id'],status='failed',facts={'error':{'stage':'model','type':'CancelledError','code':'process_interrupted','message':'emotion analysis cancelled'}})
            except Exception:
                logging.getLogger(__name__).warning("Unable to persist cancelled emotion analysis %s", state["analysis_id"])
            raise
        except Exception as exc:
            return {'error':error_facts(exc,stage='model',secret=secret)}
        if outcome.error:
            return {'outcome':outcome,'error':safe_facts(outcome.error,secret)}
        try:
            result=parse_result(outcome.raw_output or '',runtime.taxonomy)
            return {'outcome':outcome,'result':result.to_dict()}
        except Exception as exc:
            return {'outcome':outcome,'error':error_facts(exc,stage='validation' if isinstance(exc,ResultValidationError) else 'parse',secret=secret)}

    async def persist(state):
        if state.get('replay'): return {}
        facts={'latency_ms':int((monotonic()-state['started'])*1000)}
        outcome=state.get('outcome')
        if outcome:
            facts.update(raw_output=outcome.raw_output,reasoning_content=outcome.reasoning_content,
                metadata=outcome.metadata,finish_reason=outcome.finish_reason,**asdict(outcome.usage))
        if state.get('error'):
            facts['error']=state['error'];status='failed'
        else:
            facts['result']=state['result'];status='completed'
        await repo.finish(state['analysis_id'],status=status,facts=safe_facts(facts,secret))
        return {'analysis_status':status}

    builder=StateGraph(AnalysisState)
    builder.add_node('prepare_analysis',prepare)
    builder.add_node('call_emotion_model',call)
    builder.add_node('persist_analysis',persist)
    builder.add_edge(START,'prepare_analysis')
    builder.add_edge('prepare_analysis','call_emotion_model')
    builder.add_edge('call_emotion_model','persist_analysis')
    builder.add_edge('persist_analysis',END)
    return builder.compile()
