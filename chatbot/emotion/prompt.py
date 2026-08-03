"""Prompt construction for emotion recognition."""

from typing import Any

from chatbot.emotion.examples import EmotionExample, select_emotion_examples
from chatbot.emotion.labels import format_emotion_label_guidance
from chatbot.core.prompt_config import DEFAULT_EMOTION_ANALYSIS_PROMPT, load_prompt_config


def build_emotion_analysis_prompt(
    *,
    emotion_labels: list[str],
    emotion_label_set: set[str],
    dialogue_context: str,
    current_input: str = "",
    previous_emotion: str = "",
    likely_emotions: list[str] | None = None,
    examples: list[dict[str, Any]] | None = None,
    include_static_examples: bool = True,
) -> str:
    """Build a few-shot emotion-recognition prompt."""
    normalized_likely = normalize_likely_emotions(
        emotion_label_set,
        likely_emotions=likely_emotions,
        previous_emotion=previous_emotion,
    )
    likely_line = ""
    if normalized_likely:
        likely_line = f"\n- More likely emotion labels: {', '.join(normalized_likely)}"

    example_block = _render_selected_examples(examples)
    if not example_block and include_static_examples:
        example_block = format_emotion_examples(
            select_emotion_examples(
                likely_emotions=normalized_likely,
                current_input=current_input,
            )
        )

    labels = ", ".join(emotion_labels)
    values = {
        "emotion_labels": labels,
        "label_guidance": format_emotion_label_guidance(emotion_labels),
        "example_block": example_block,
        "likely_line": likely_line,
        "dialogue_context": dialogue_context,
    }
    template = load_prompt_config().emotion_analysis
    try:
        return template.format(**values).strip()
    except (KeyError, IndexError, ValueError):
        return DEFAULT_EMOTION_ANALYSIS_PROMPT.format(**values).strip()


def normalize_likely_emotions(
    emotion_label_set: set[str],
    *,
    likely_emotions: list[str] | None = None,
    previous_emotion: str = "",
) -> list[str]:
    normalized = []
    for emotion in [previous_emotion, *(likely_emotions or [])]:
        emotion = emotion.strip().lower()
        if emotion and emotion in emotion_label_set and emotion not in normalized:
            normalized.append(emotion)
    return normalized


def format_emotion_examples(examples: list[EmotionExample]) -> str:
    if not examples:
        return ""
    lines = ["", "Labeled examples:"]
    for index, example in enumerate(examples, start=1):
        lines.append(f"{index}. Dialogue example: {example.dialogue}")
        lines.append(f"   True emotion label: {example.emotion}")
    return "\n".join(lines)


def _render_selected_examples(examples: list[dict[str, Any]] | None) -> str:
    if not examples:
        return ""
    lines = ["", "Dynamic EICL examples:"]
    for example in examples:
        score = float(example.get("score", 0.0))
        lines.append(f"- Dialogue: {example.get('dialogue', '')}")
        lines.append(f"  True emotion label: {example.get('emotion', '')}")
        lines.append(
            f"  Selection reason: {example.get('reason', '')} (score={score:.2f})"
        )
    return "\n".join(lines)
