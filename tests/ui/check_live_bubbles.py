"""Explicit opt-in checks of artificial inputs against the configured chat model."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import dotenv_values

from chatbot.core.config import AppConfig
from chatbot.llm.bubbles import BubbleStream
from chatbot.llm.openai_compatible import OpenAICompatibleChatModel
from chatbot.llm.prompt import build_prompt


INPUTS = [
    "今天下班终于赶上了一班空地铁，整个人都舒坦了。",
    "最近每天忙到很晚，但总觉得事情没做好。今天又被指出几个问题，有点泄气，也不想马上听一堆建议，想找个人说说。",
    "请具体解释一下 Python 列表和元组的区别，包括可变性、使用场景，再给一个完整的代码例子。",
]


async def check():
    config = AppConfig.from_env()
    model = OpenAICompatibleChatModel(config)
    results = []
    try:
        for content in INPUTS:
            reply = BubbleStream()
            start = time.monotonic()
            first = None
            error_type = None
            try:
                async for delta in model.stream(build_prompt([{"role": "user", "content": content}])):
                    if error_type is None:
                        try:
                            reply.feed(delta.content)
                            if reply.bubbles and first is None:
                                first = round((time.monotonic() - start) * 1000)
                        except ValueError as error:
                            error_type = type(error).__name__
                if error_type is None:
                    reply.finish()
                    if reply.mode != "json":
                        error_type = "ExpectedJSONMode"
            except Exception as error:
                error_type = type(error).__name__
            result = {"input": content, "bubbles": reply.bubbles, "format": reply.mode,
                      "first_bubble_ms": first, "elapsed_ms": round((time.monotonic() - start) * 1000),
                      "error_type": error_type}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        await model.client.root_async_client.close()
        model.client.root_client.close()
    target = Path(__file__).resolve().parents[2] / "docs/verification/chat-bubbles/live.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"model": config.llm_model, "results": results}, ensure_ascii=False, indent=2) + "\n")
    if any(item["error_type"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True, help="Read credentials without copying or printing them")
    args = parser.parse_args()
    for key, value in dotenv_values(args.env_file).items():
        if value is not None:
            os.environ.setdefault(key, value)
    asyncio.run(check())
