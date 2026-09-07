"""Isolated browser fixture: real app, deterministic replies, no external calls."""

import asyncio
import json
from pathlib import Path

from pydantic import SecretStr

from chatbot.core.config import AppConfig
from chatbot.llm.types import ModelDelta
from tests.emotion.helpers import create_app


class Chat:
    parameters = {"model": "offline-bubbles-ui"}

    async def stream(self, messages):
        text = str(messages[-1].content)
        yield ModelDelta(content='{"messages":[' + json.dumps('今天确实够忙的。', ensure_ascii=False) + ',')
        await asyncio.sleep(1.2)
        if "失败" in text:
            raise RuntimeError("offline failure after one complete bubble")
        yield ModelDelta(content=json.dumps("忙完了就先让自己歇一会儿。\n不急着把剩下的事情都安排上。", ensure_ascii=False) + ",")
        yield ModelDelta(content=json.dumps("要是想聊聊，我在听。", ensure_ascii=False) + "]}")


def build_app():
    config = AppConfig(SecretStr("offline"), "offline-ui", None, 0, 10, 40,
                       Path(__file__).resolve().parents[2] / "data/bubbles-ui.sqlite3")
    return create_app(config=config, model=Chat())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_app(), host="127.0.0.1", port=8766, access_log=False)
