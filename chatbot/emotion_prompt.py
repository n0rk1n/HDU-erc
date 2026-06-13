"""Prompt construction for emotion recognition."""

from chatbot.emotion_examples import EmotionExample, select_emotion_examples


def build_emotion_analysis_prompt(
    *,
    emotion_labels: list[str],
    emotion_label_set: set[str],
    dialogue_context: str,
    current_input: str = "",
    previous_emotion: str = "",
    likely_emotions: list[str] | None = None,
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

    examples = format_emotion_examples(
        select_emotion_examples(
            likely_emotions=normalized_likely,
            current_input=current_input,
        )
    )

    labels = ", ".join(emotion_labels)
    return f"""Infer the user's current emotion from the dialogue context.
- Dialogue context: The conversation history between user and assistant, with utterances separated by </s>.
- Emotion labels: {labels}
- Choose a single inferred emotion from the provided Emotion labels, not outside of them.
- Response Format: Emotion: [a single inferred emotion]{likely_line}
{examples}

Dialogue context: {dialogue_context}""".strip()


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
