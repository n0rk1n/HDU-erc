"""Incrementally decode complete strings from the model's JSON reply array."""

from __future__ import annotations

import json
import re


REPLY_FORMAT_PROMPT = """回复传输格式（必须遵守，不向用户解释）：
仅输出一个 JSON 对象，格式为 {"messages":["完整气泡正文"]}。
顶层必须是对象，不能直接输出字符串数组、纯文本或其他 JSON 类型。
messages 是非空字符串数组，通常包含 1–3 个自然的聊天气泡。每个气泡可以有几句话或完整段落。
不要为了凑数拆分内容。正文中的换行、引号和代码按 JSON 字符串转义。
不要在 JSON 外输出文字或包裹 Markdown 代码围栏，不要添加其他字段。
用户需要代码、列表或详细解释时，把完整内容放在字符串中，仍然使用同一传输格式。
示例：{"messages":["今天挺熬人的啊。","想聊什么都行，我在听。"]}"""

_PREFIX = re.compile(r'^\s*\{\s*"messages"\s*:\s*\[')


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate reply field")
        result[key] = value
    return result


class BubbleStream:
    """Publish complete strings early, preserving legacy plain-text responses.

    First two bubbles are ready as their strings close. The final bubble waits
    for full validation at EOF; excess items are combined without truncation.
    """

    def __init__(self, *, require_object: bool = False) -> None:
        self.require_object = require_object
        self.raw_output = ""
        self.mode: str | None = None
        self.bubbles: list[str] = []
        self._texts: list[str] = []
        self._cursor: int | None = None
        self._expect_value = True
        self._closed = False
        self._decoder = json.JSONDecoder()

    @property
    def content(self) -> str:
        return self.raw_output if self.mode == "plain" else "\n\n".join(self.bubbles)

    def feed(self, text: str) -> None:
        self.raw_output += text
        if self.mode is None and self.raw_output.strip():
            first = self.raw_output.lstrip()[0]
            self.mode = "json" if self.require_object or first in '{[`' else "plain"
        if self.mode != "json" or self._closed:
            return
        if self._cursor is None:
            prefix = _PREFIX.match(self.raw_output)
            if prefix is None:
                return  # Incomplete headers are validated at EOF, never displayed.
            self._cursor = prefix.end()
        while self._cursor < len(self.raw_output):
            char = self.raw_output[self._cursor]
            if char.isspace():
                self._cursor += 1
                continue
            if char == ']':
                self._closed = True
                return
            if not self._expect_value:
                if char != ',':
                    raise ValueError("invalid bubble separator")
                self._cursor += 1
                self._expect_value = True
                continue
            try:
                value, end = self._decoder.raw_decode(self.raw_output, self._cursor)
            except json.JSONDecodeError:
                return  # A JSON string can span arbitrary provider chunks.
            if not isinstance(value, str) or not value.strip():
                raise ValueError("invalid reply bubble")
            self._texts.append(value)
            if len(self.bubbles) < 2:
                self.bubbles.append(value)
            self._cursor = end
            self._expect_value = False

    def finish(self) -> None:
        if self.mode == "plain":
            self.bubbles = [self.raw_output]
            return
        value = json.loads(self.raw_output, object_pairs_hook=_unique_fields)
        if (not isinstance(value, dict) or set(value) != {"messages"}
                or not isinstance(value["messages"], list) or not self._texts
                or value["messages"] != self._texts):
            raise ValueError("invalid reply object")
        if len(self._texts) > 2:
            self.bubbles.append("\n\n".join(self._texts[2:]))
