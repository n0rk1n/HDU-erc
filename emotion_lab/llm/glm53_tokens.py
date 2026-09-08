"""Offline GLM-5.3 text-message counting with pinned official assets."""

from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from emotion_lab.config import ConfigError

REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
TOKENIZER_PATH = Path(__file__).parent / "vendor/glm53/tokenizer.json"
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"


@lru_cache(maxsize=1)
def _load_tokenizer():
    try:
        from tokenizers import Tokenizer

        data = TOKENIZER_PATH.read_bytes()
        if sha256(data).hexdigest() != TOKENIZER_SHA256:
            raise ValueError("tokenizer checksum mismatch")
        return Tokenizer.from_str(data.decode("utf-8"))
    except (ImportError, OSError, ValueError) as exc:
        raise ConfigError(
            "cannot load bundled GLM-5.3 tokenizer; restore official assets"
        ) from exc


def _text_payload(messages):
    roles = {"system": "system", "human": "user", "ai": "assistant"}
    payload = []
    for message in messages:
        if (
            message.type not in roles
            or not isinstance(message.content, str)
            or getattr(message, "tool_calls", None)
            or getattr(message, "invalid_tool_calls", None)
            or set(message.additional_kwargs) - {"reasoning_content"}
        ):
            raise ConfigError(
                "GLM-5.3 emotion counter supports text-only messages without tools"
            )
        reasoning = message.additional_kwargs.get("reasoning_content")
        if reasoning is not None and (
            message.type != "ai" or not isinstance(reasoning, str)
        ):
            raise ConfigError(
                "GLM-5.3 emotion counter requires text-only assistant reasoning"
            )
        payload.append((roles[message.type], message.content, reasoning))
    return payload


def _render_text(payload, effort, clear_thinking):
    """The text-only branch of vendor/glm53/chat_template.jinja, without tools."""
    parts = [f"[gMASK]<sop><|system|>Reasoning Effort: {effort}"]
    last_user = max(
        (i for i, (role, _, _) in enumerate(payload) if role == "user"), default=-1
    )
    for index, (role, content, reasoning) in enumerate(payload):
        parts.append(f"<|{role}|>")
        if role == "assistant":
            if reasoning is None and "</think>" in content:
                reasoning = content.split("</think>")[0].split("<think>")[-1]
                content = content.split("</think>")[-1]
            if clear_thinking and index <= last_user:
                reasoning = None
            parts.extend(("<think>", reasoning or "", "</think>", content.strip()))
        else:
            parts.append(content)
    parts.append("<|assistant|><think>")
    return "".join(parts)


class GLM53TokenCounter:
    identity = "ZHIPU/GLM-5.3"

    def __init__(self):
        self.tokenizer = _load_tokenizer()
        self.version = (
            f"glm-5.3:{REVISION};tokenizers:{version('tokenizers')};text-template-v1"
        )

    def count(self, messages):
        if not messages:
            return 0
        payload = _text_payload(messages)
        # Cover all supported efforts and provider choices about retained reasoning.
        # The adapters send no tools or response schema; never estimate those here.
        return max(
            len(
                self.tokenizer.encode(
                    _render_text(payload, effort, clear), add_special_tokens=False
                ).ids
            )
            for effort in ("Low", "High", "Max")
            for clear in (False, True)
        )
