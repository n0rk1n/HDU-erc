"""Few-shot examples for emotion recognition prompts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionExample:
    dialogue: str
    emotion: str


DEFAULT_EMOTION_EXAMPLES = [
    EmotionExample(
        dialogue="I keep thinking about the interview tomorrow</s>I can barely sit still.",
        emotion="anxious",
    ),
    EmotionExample(
        dialogue="I thought they would remember my birthday</s>Nobody said anything all day.",
        emotion="disappointed",
    ),
    EmotionExample(
        dialogue="I finally finished the project</s>It took weeks, but I did it.",
        emotion="proud",
    ),
    EmotionExample(
        dialogue="That message from my old friend made me smile</s>I miss those days.",
        emotion="nostalgic",
    ),
    EmotionExample(
        dialogue="Thank you for staying with me through this</s>It really means a lot.",
        emotion="grateful",
    ),
    EmotionExample(
        dialogue="I do not want to bother anyone</s>It feels like I am handling this alone.",
        emotion="lonely",
    ),
    EmotionExample(
        dialogue="I cannot believe they said that to me</s>I am shaking just thinking about it.",
        emotion="angry",
    ),
    EmotionExample(
        dialogue="The results came back better than I expected</s>I feel light again.",
        emotion="joyful",
    ),
]


def select_emotion_examples(
    *,
    likely_emotions: list[str] | None = None,
    current_input: str = "",
    limit: int = 3,
) -> list[EmotionExample]:
    """Select a small deterministic few-shot set, preferring likely emotions."""
    selected: list[EmotionExample] = []
    normalized_likely = [
        emotion.strip().lower()
        for emotion in (likely_emotions or [])
        if emotion.strip()
    ]

    for emotion in normalized_likely:
        for example in DEFAULT_EMOTION_EXAMPLES:
            if example.emotion == emotion and example not in selected:
                selected.append(example)
                break
        if len(selected) >= limit:
            return selected

    input_text = current_input.lower()
    for example in DEFAULT_EMOTION_EXAMPLES:
        if example in selected:
            continue
        if example.emotion in input_text:
            selected.append(example)
        if len(selected) >= limit:
            return selected

    for example in DEFAULT_EMOTION_EXAMPLES:
        if example not in selected:
            selected.append(example)
        if len(selected) >= limit:
            break

    return selected
