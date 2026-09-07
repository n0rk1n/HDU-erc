import json

import pytest

from chatbot.llm.bubbles import BubbleStream


def test_protocol_boundaries_survive_every_chunk_split_and_preserve_multiline_code():
    """Catches chunk-dependent splitting, damaged escaping, or code becoming many bubbles."""
    expected = ['嗯，我听着。', '可以这样写：\n```python\nprint("你好")\n```\n\n这段说明也在同一条。']
    raw = json.dumps({'messages': expected}, ensure_ascii=False, indent=2)
    for boundary in range(len(raw) + 1):
        stream = BubbleStream()
        stream.feed(raw[:boundary])
        stream.feed(raw[boundary:])
        stream.finish()
        assert stream.bubbles == expected
        assert stream.content == '\n\n'.join(expected)


def test_plain_reply_is_one_complete_bubble_even_with_paragraphs():
    """Catches turning ordinary paragraph breaks into extra chat messages."""
    stream = BubbleStream()
    stream.feed('第一段。\n\n')
    assert stream.bubbles == []
    stream.feed('第二段比较长，也应完整保留。')
    stream.finish()
    assert stream.bubbles == ['第一段。\n\n第二段比较长，也应完整保留。']


def test_excess_bubbles_merge_into_last_bubble_without_truncating_content():
    """Catches flooding the UI or losing explanations when a model exceeds three objects."""
    stream = BubbleStream()
    stream.feed('{"messages":["一","二","三","四"]}')
    assert stream.bubbles == ['一', '二']
    stream.finish()
    assert stream.bubbles == ['一', '二', '三\n\n四']


@pytest.mark.parametrize('raw', [
    '', '  ', '{"messages":["unfinished', '{"messages":[""]}', '{"messages":[2]}',
    '{"messages":[],"extra":true}', '{"messages":[],"messages":[]}',
    '["a"]', '```json\n{"text":"a"}\n```',
])
def test_invalid_structured_output_is_not_displayed_as_chat(raw):
    """Catches protocol text or malformed output leaking into visible history."""
    stream = BubbleStream()
    with pytest.raises(ValueError):
        stream.feed(raw)
        stream.finish()
    assert stream.bubbles == []


def test_valid_bubble_before_invalid_line_is_retained_even_in_the_same_chunk():
    stream = BubbleStream()
    with pytest.raises(ValueError):
        stream.feed('{"messages":["已完成",false]}')
    assert stream.bubbles == ['已完成']
    assert stream.content == '已完成'
