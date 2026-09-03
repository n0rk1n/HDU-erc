"""Pure prompt, parsing, and one-shot emotion analysis helpers."""

import re
from dataclasses import dataclass
from typing import Any

from chatbot.emotion.examples import DEFAULT_EMOTION_EXAMPLES
from chatbot.emotion.labels import EMOTION_LABELS, EMOTION_LABEL_SET
from chatbot.emotion.prompt import build_emotion_analysis_prompt
from chatbot.emotion.retrieval import select_dynamic_examples
from chatbot.emotion.state import EmotionState, emotion_state_from_output


@dataclass(frozen=True)
class EmotionAnalysisResult:
    emotion: str
    input: str
    output: str
    success: bool
    error: str = ""
    state: EmotionState | None = None


def _recent_contents(records: list[dict], max_turns: int) -> list[str]:
    limit = max(1, max_turns) * 2
    recent = records[-limit:]
    return [
        str(record.get("content", "")).strip()
        for record in recent
        if str(record.get("content", "")).strip()
    ]


def _dialogue_context(records: list[dict], current_input: str, max_turns: int) -> str:
    utterances = _recent_contents(records, max_turns)
    current_input = current_input.strip()
    if current_input:
        utterances.append(current_input)
    return "</s>".join(utterances)


def build_emotion_prompt(
    records: list[dict],
    current_input: str,
    *,
    previous_emotion: str = "",
    likely_emotions: list[str] | None = None,
    max_turns: int = 5,
    example_mode: str = "dynamic",
    include_emotion_history: bool = True,
    prompt_variant: str = "full",
) -> str:
    dialogue_context = _dialogue_context(records, current_input, max_turns)

    if example_mode not in {"dynamic", "static", "none"}:
        raise ValueError("example_mode must be one of: dynamic, static, none.")

    prompt_previous_emotion = previous_emotion if include_emotion_history else ""
    prompt_likely_emotions = likely_emotions if include_emotion_history else None
    retrieval_likely_emotions = [
        emotion
        for emotion in [prompt_previous_emotion, *(prompt_likely_emotions or [])]
        if emotion
    ]

    selected_examples = None
    if example_mode == "dynamic":
        selected_examples = select_dynamic_examples(
            examples=DEFAULT_EMOTION_EXAMPLES,
            dialogue_context=dialogue_context,
            likely_emotions=retrieval_likely_emotions,
            limit=4,
        )

    return build_emotion_analysis_prompt(
        emotion_labels=EMOTION_LABELS,
        emotion_label_set=EMOTION_LABEL_SET,
        dialogue_context=dialogue_context,
        current_input=current_input,
        previous_emotion=prompt_previous_emotion,
        likely_emotions=prompt_likely_emotions,
        examples=selected_examples,
        include_static_examples=example_mode in {"dynamic", "static"},
        prompt_variant=prompt_variant,
    )


def parse_emotion_output(output: str) -> str | None:
    """从 LLM 输出中提取情绪标签，并校验其在预定义标签集中。"""
    match = re.search(r"Emotion:\s*([A-Za-z_-]+)", output)
    if not match:
        return None
    emotion = match.group(1).strip().lower()
    if emotion not in EMOTION_LABEL_SET:
        return None
    return emotion


def _response_output(response: Any) -> str:
    content = response.content if hasattr(response, "content") else str(response)
    return content if isinstance(content, str) else str(content)


def _result_from_output(prompt: str, output: str) -> EmotionAnalysisResult:
    state = emotion_state_from_output(output)
    emotion = state.primary_emotion if state else parse_emotion_output(output)
    if emotion is None:
        return EmotionAnalysisResult(
            "",
            prompt,
            output,
            False,
            "Failed to parse a known emotion label.",
        )
    if state is None:
        state = EmotionState(primary_emotion=emotion)
    return EmotionAnalysisResult(emotion, prompt, output, True, state=state)


async def analyze_emotion_async(
    llm,
    records: list[dict],
    current_input: str,
    *,
    previous_emotion: str = "",
    likely_emotions: list[str] | None = None,
    turn_count: int,
    emotion_interval: int,
) -> EmotionAnalysisResult:
    """Build and analyze one emotion prompt without persistence side effects."""
    prompt = build_emotion_prompt(
        records,
        current_input,
        previous_emotion=previous_emotion,
        likely_emotions=likely_emotions,
        max_turns=emotion_interval,
    )
    try:
        output = _response_output(await llm.ainvoke(prompt))
        return _result_from_output(prompt, output)
    except Exception as exc:
        return EmotionAnalysisResult("", prompt, "", False, str(exc))
