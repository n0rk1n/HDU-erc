from urllib.parse import urlsplit
from emotion_lab.config import ConfigError


def thinking_extra_body(
    model: str,
    base_url: str | None,
    thinking: str,
    reasoning_effort: str | None = None,
) -> dict | None:
    """Apply the supported provider's thinking contract to every model role."""
    host = urlsplit(base_url or "").hostname or ""
    if host == "api.deepseek.com" and model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        return {"thinking": {"type": thinking}}
    if model == "ZHIPU/GLM-5.3" and (
        host == "dashscope.aliyuncs.com"
        or host.endswith(".cn-beijing.maas.aliyuncs.com")
    ):
        if thinking != "enabled":
            raise ConfigError("GLM-5.3 requires thinking enabled")
        if reasoning_effort not in {None, "low", "high", "max"}:
            raise ConfigError("GLM-5.3 reasoning effort must be low, high or max")
        return {
            "enable_thinking": True,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        }
    return None
