"""Explicit project-relative paths; credentials never enter experiment snapshots."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    pass


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_environment(path=None):
    # No find_dotenv(): never search a parent checkout.
    path = project_path(path) if path else PROJECT_ROOT / ".env"
    if path.exists():
        from dotenv import load_dotenv

        load_dotenv(path, override=False)


def credentials():
    key = os.getenv("EMOTION_LLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise ConfigError(
            "configure EMOTION_LLM_API_KEY or LLM_API_KEY in the task worktree"
        )
    return key


def validate_config(config):
    config = json.loads(json.dumps(config, allow_nan=False))
    allowed = {
        "method",
        "seed",
        "k",
        "candidate_count",
        "max_attempts",
        "concurrency",
        "repetition_no",
        "name",
        "parser_version",
        "deduplicate",
        "require_evidence",
        "max_total_attempts",
        "model",
        "pricing",
        "corpus_set",
        "evaluation_set",
        "corpus_set_id",
        "evaluation_set_id",
        "comparison_group",
        "research_question",
        "prompt_version",
        "example_policy",
    }
    if set(config) - allowed:
        raise ConfigError(
            "unknown run configuration keys: " + ",".join(sorted(set(config) - allowed))
        )
    if config.get("method") not in {"zero-shot", "random", "lexical"}:
        raise ConfigError(
            "method not implemented; supported: zero-shot, random, lexical"
        )
    for key, default, minimum, maximum in [
        ("seed", 42, 0, 2**32),
        ("k", 0, 0, 100),
        ("candidate_count", 50, 1, 10000),
        ("max_attempts", 2, 1, 10),
        ("concurrency", 1, 1, 16),
        ("repetition_no", 1, 1, 100000),
    ]:
        value = config.setdefault(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ConfigError(f"invalid {key}")
    if config["method"] == "zero-shot" and config["k"] != 0:
        raise ConfigError("zero-shot requires k=0")
    if (
        config["method"] != "zero-shot"
        and not 0 < config["k"] <= config["candidate_count"]
    ):
        raise ConfigError("retrieval requires 0 < k <= candidate_count")
    config.setdefault("name", "GoEmotions experiment")
    config.setdefault("parser_version", "labels-v1")
    config.setdefault("deduplicate", True)
    config.setdefault("require_evidence", False)
    config.setdefault("prompt_version", "native-labels-v1")
    config.setdefault("example_policy", "ranked")
    if config["prompt_version"] not in {"native-labels-v1", "native-labels-v2", "native-labels-v3", "native-labels-v4"}:
        raise ConfigError("unsupported prompt_version")
    if config["example_policy"] not in {"ranked", "contrastive-v1"}:
        raise ConfigError("unsupported example_policy")
    if config["example_policy"] == "contrastive-v1" and (
        config["method"] != "lexical" or config["k"] < 2
    ):
        raise ConfigError("contrastive-v1 requires lexical retrieval with k >= 2")
    config.setdefault("max_total_attempts", 100000)
    if (
        type(config["max_total_attempts"]) is not int
        or config["max_total_attempts"] < 1
    ):
        raise ConfigError("max_total_attempts must be positive")
    model = config.setdefault("model", {})

    def env(field, default):
        return os.getenv("EMOTION_LLM_" + field) or os.getenv("LLM_" + field) or default

    defaults = {
        "name": os.getenv("EMOTION_LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or "deepseek-v4-flash",
        "base_url": os.getenv("EMOTION_LLM_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://api.deepseek.com",
        "temperature": float(env("TEMPERATURE", "0")),
        "timeout_seconds": float(env("TIMEOUT_SECONDS", "120")),
        "max_tokens": int(os.getenv("EMOTION_OUTPUT_TOKENS") or "4096"),
        "context_tokens": int(os.getenv("EMOTION_CONTEXT_TOKENS") or "32768"),
        "safety_tokens": int(os.getenv("EMOTION_SAFETY_TOKENS") or "256"),
        "tokenizer_model": os.getenv("EMOTION_TOKENIZER_MODEL") or "deepseek-v4-flash",
        "thinking": env("THINKING", "disabled"),
    }
    for key, value in defaults.items():
        model.setdefault(key, value)
    effort = env("REASONING_EFFORT", None)
    if effort:
        model.setdefault("reasoning_effort", effort)
    if set(model) - set(defaults) - {"reasoning_effort"}:
        raise ConfigError(
            "unsupported model configuration (credentials must stay in environment)"
        )
    url = urlsplit(model["base_url"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ConfigError(
            "base_url must be an HTTP endpoint without credentials, query or fragment"
        )
    import math

    if (
        isinstance(model["timeout_seconds"], bool)
        or not isinstance(model["timeout_seconds"], (int, float))
        or not math.isfinite(model["timeout_seconds"])
        or model["timeout_seconds"] <= 0
    ):
        raise ConfigError("invalid timeout")
    if type(model["safety_tokens"]) is not int or model["safety_tokens"] < 0:
        raise ConfigError("invalid safety token margin")
    for key in ("max_tokens", "context_tokens"):
        if type(model[key]) is not int or model[key] <= 0:
            raise ConfigError(f"invalid model {key}")
    if model["max_tokens"] + model["safety_tokens"] >= model["context_tokens"]:
        raise ConfigError("output budget exceeds context")
    if (
        isinstance(model["temperature"], bool)
        or not isinstance(model["temperature"], (int, float))
        or not 0 <= model["temperature"] <= 2
    ):
        raise ConfigError("temperature must be within 0..2")
    if model["thinking"] not in {"enabled", "disabled"}:
        raise ConfigError("invalid thinking setting")
    pricing = config.get("pricing")
    if pricing is not None:
        from decimal import Decimal, InvalidOperation

        if not isinstance(pricing, dict) or not {
            "currency",
            "input_per_million",
            "output_per_million",
        } <= set(pricing):
            raise ConfigError(
                "pricing requires currency and input/output rates per million tokens"
            )
        if not isinstance(pricing["currency"], str) or not pricing["currency"]:
            raise ConfigError("pricing currency is required")
        for key in (
            "input_per_million",
            "output_per_million",
            "cached_input_per_million",
        ):
            if key in pricing:
                try:
                    rate = Decimal(str(pricing[key]))
                except InvalidOperation:
                    raise ConfigError("invalid price rate") from None
                if not rate.is_finite() or rate < 0:
                    raise ConfigError("invalid price rate")
    return config
