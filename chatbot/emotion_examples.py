"""Few-shot examples for emotion recognition prompts."""

import json
from dataclasses import dataclass
from pathlib import Path

from chatbot.emotion_labels import EMOTION_LABEL_SET


@dataclass(frozen=True)
class EmotionExample:
    dialogue: str
    emotion: str
    example_id: str = ""
    source_split: str = ""


FALLBACK_EMOTION_EXAMPLES = [
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

DEFAULT_EXAMPLE_BANK_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "benchmarks"
    / "empathetic_dialogues_v1"
    / "few_shot"
    / "train_examples.jsonl"
)


def load_emotion_examples(path: Path = DEFAULT_EXAMPLE_BANK_PATH) -> list[EmotionExample]:
    """Load the human-authored train bank, with a small offline fallback."""
    if not path.exists():
        return list(FALLBACK_EMOTION_EXAMPLES)
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid example JSONL at {path}:{line_number}") from exc
        dialogue = str(item.get("dialogue", "")).strip()
        emotion = str(item.get("emotion", "")).strip().lower()
        if not dialogue or emotion not in EMOTION_LABEL_SET:
            raise ValueError(f"Invalid emotion example at {path}:{line_number}")
        examples.append(
            EmotionExample(
                dialogue=dialogue,
                emotion=emotion,
                example_id=str(item.get("example_id", "")),
                source_split=str(item.get("source_split", "")),
            )
        )
    return examples or list(FALLBACK_EMOTION_EXAMPLES)


DEFAULT_EMOTION_EXAMPLES = load_emotion_examples()


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
