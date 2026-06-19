"""Dynamic example retrieval for emotion recognition prompts."""

import re
from typing import Any

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def select_dynamic_examples(
    *,
    examples: list[Any],
    dialogue_context: str,
    likely_emotions: list[str] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    context_tokens = _tokenize(dialogue_context)
    likely_set = {
        emotion.strip().lower()
        for emotion in (likely_emotions or [])
        if emotion.strip()
    }
    ranked = [
        _score_example(
            index=index,
            example=example,
            context_tokens=context_tokens,
            likely_emotions=likely_set,
        )
        for index, example in enumerate(examples)
    ]
    ranked.sort(key=lambda item: (-item["score"], item["index"]))

    return _select_distinct_emotions(ranked, max(1, limit))


def _score_example(
    *,
    index: int,
    example: Any,
    context_tokens: set[str],
    likely_emotions: set[str],
) -> dict[str, Any]:
    dialogue = _example_value(example, "dialogue")
    emotion = _example_value(example, "emotion").strip().lower()
    overlap = sorted(context_tokens & _tokenize(dialogue))
    boosted = emotion in likely_emotions
    score = float(len(overlap))
    if boosted:
        score += 2.0

    reasons = []
    if overlap:
        reasons.append(f"overlap={','.join(overlap)}")
    if boosted:
        reasons.append("recent-emotion-prior")
    if not reasons:
        reasons.append("fallback-diversity")

    return {
        "index": index,
        "dialogue": dialogue,
        "emotion": emotion,
        "score": score,
        "reason": "; ".join(reasons),
    }


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def _example_value(example: Any, name: str) -> str:
    if isinstance(example, dict):
        return str(example.get(name, ""))
    return str(getattr(example, name, ""))


def _select_distinct_emotions(
    ranked: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected = []
    seen_emotions = set()
    for item in ranked:
        if item["emotion"] in seen_emotions:
            continue
        selected.append(item)
        seen_emotions.add(item["emotion"])
        if len(selected) >= limit:
            return selected

    for item in ranked:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break

    return selected
