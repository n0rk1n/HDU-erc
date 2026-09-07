"""Offline V4 text-message counting with DeepSeek's pinned official assets."""
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from chatbot.core.errors import ConfigError
from chatbot.llm.vendor.deepseek_v4.encoding_dsv4 import encode_messages

REVISION = '60d8d70770c6776ff598c94bb586a859a38244f1'
TOKENIZER_PATH = Path(__file__).parent / 'vendor/deepseek_v4/tokenizer.json'
TOKENIZER_SHA256 = '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf'


@lru_cache(maxsize=1)
def _load_tokenizer():
    try:
        from tokenizers import Tokenizer
        data = TOKENIZER_PATH.read_bytes()
        if sha256(data).hexdigest() != TOKENIZER_SHA256:
            raise ValueError('tokenizer checksum mismatch')
        return Tokenizer.from_str(data.decode('utf-8'))
    except (ImportError, OSError, ValueError) as exc:
        raise ConfigError('cannot load bundled DeepSeek tokenizer; reinstall requirements and restore official assets') from exc


class DeepSeekV4TokenCounter:
    identity = 'deepseek-v4-flash'

    def __init__(self):
        self.tokenizer = _load_tokenizer()
        self.version = f'deepseek-v4:{REVISION};tokenizers:{version("tokenizers")};text-template-v1'

    def count(self, messages):
        if not messages:
            return 0
        roles = {'system': 'system', 'human': 'user', 'ai': 'assistant'}
        payload = []
        for message in messages:
            if (message.type not in roles or not isinstance(message.content, str)
                    or getattr(message, 'tool_calls', None)
                    or getattr(message, 'invalid_tool_calls', None)
                    or set(message.additional_kwargs) - {'reasoning_content'}):
                raise ConfigError('DeepSeek emotion counter supports text-only messages without tools')
            item = {'role': roles[message.type], 'content': message.content}
            reasoning = message.additional_kwargs.get('reasoning_content')
            if reasoning is not None:
                if message.type != 'ai' or not isinstance(reasoning, str):
                    raise ConfigError('DeepSeek emotion counter requires text-only assistant reasoning')
                item['reasoning_content'] = reasoning
            payload.append(item)
        # No tools or response schema are sent by the emotion/gate adapters.
        # Cover both provider thinking defaults without relying on an OpenAI template.
        return max(len(self.tokenizer.encode(
            encode_messages(payload, thinking_mode=mode), add_special_tokens=False).ids)
            for mode in ('chat', 'thinking'))
