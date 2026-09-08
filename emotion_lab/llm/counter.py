from types import SimpleNamespace
from emotion_lab.config import ConfigError


class TokenCounter:
    def __init__(self, model):
        if model == "deepseek-v4-flash":
            from .deepseek_tokens import DeepSeekV4TokenCounter

            self.counter = DeepSeekV4TokenCounter()
        elif model == "ZHIPU/GLM-5.3":
            from .glm53_tokens import GLM53TokenCounter

            self.counter = GLM53TokenCounter()
        else:
            raise ConfigError(
                "unsupported tokenizer; supply a verified matching counter"
            )
        self.identity = self.counter.identity
        self.version = self.counter.version

    def count(self, messages):
        roles = {"system": "system", "user": "human", "assistant": "ai"}
        return self.counter.count(
            [
                SimpleNamespace(
                    type=roles[m["role"]], content=m["content"], additional_kwargs={}
                )
                for m in messages
            ]
        )
