"""Offline browser QA server. Run as a module; never calls an external model."""
import asyncio
import json
from pathlib import Path

from pydantic import SecretStr

from chatbot.core.config import AppConfig
from chatbot.db.connection import Database
from chatbot.emotion.types import ModelOutcome
from chatbot.llm.types import ModelDelta, TokenUsage
from tests.emotion.helpers import create_app, emotion_runtime
from tests.emotion_gate.helpers import gate_runtime


class Emotion:
    parameters = {'model': 'offline-ui', 'max_retries': 0}

    async def invoke(self, messages):
        await asyncio.sleep(2)
        text = str(messages[-1].content)
        if '识别失败' in text:
            raise RuntimeError('offline recognition failure')
        return ModelOutcome(json.dumps({
            'primary_emotion': 'anxious', 'confidence': .88, 'secondary_emotions': [],
            'evidence': '你提到担心明天的汇报，并表达了紧张的感受。',
            'reply_strategy': 'listen', 'trajectory_note': '', 'safety_level': 'normal',
        }), None, TokenUsage(), 'stop', {}, None)


class Gate:
    parameters = {'model': 'offline-ui', 'max_retries': 0}

    async def invoke(self, messages):
        await asyncio.sleep(1)
        return ModelOutcome(json.dumps({'should_analyze': '识别失败' in str(messages[-1].content),
                                      'reason': 'changed' if '识别失败' in str(messages[-1].content) else 'stable'}),
                            None, TokenUsage(), 'stop', {}, None)


class Chat:
    parameters = {'model': 'offline-ui', 'max_retries': 0}

    async def stream(self, messages):
        await asyncio.sleep(1)
        for sentence in ('明天的汇报让你有些紧张，我听到了。',
                         '\n\n如果你愿意，我们可以先从最担心的部分聊起。',
                         '是汇报内容还没准备好，还是现场表达让你感到压力？'):
            yield ModelDelta(content=sentence)
            await asyncio.sleep(.3)


def build_app():
    config = AppConfig(SecretStr('offline'), 'offline-ui', None, 0, 15, 40,
                       Path(__file__).resolve().parents[2] / 'data/ui-preview.sqlite3')
    db = Database(config.sqlite_db_path)
    return create_app(config=config, model=Chat(), emotion=emotion_runtime(db, Emotion()),
                      gate=gate_runtime(db, model=Gate()))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(build_app(), host='127.0.0.1', port=8765, access_log=False)
