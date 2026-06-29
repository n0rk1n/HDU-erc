"""情感分析模块 —— 在聊天过程中按固定轮次间隔分析用户当前情绪，结果持久化到 SQLite。"""

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chatbot.emotion_examples import DEFAULT_EMOTION_EXAMPLES
from chatbot.emotion_labels import EMOTION_LABELS, EMOTION_LABEL_SET
from chatbot.emotion_prompt import build_emotion_analysis_prompt
from chatbot.emotion_retrieval import select_dynamic_examples
from chatbot.emotion_state import EmotionState, emotion_state_from_output
from chatbot.runtime_store import DEFAULT_RUNTIME_DB_PATH, RuntimeStore

RUNTIME_DB_PATH = DEFAULT_RUNTIME_DB_PATH
EMOTION_ANALYSIS_NAMESPACE = "emotion_analysis"
_ANALYSIS_LOCK = threading.RLock()


@dataclass(frozen=True)
class EmotionAnalysisResult:
    emotion: str
    input: str
    output: str
    success: bool
    error: str = ""
    state: EmotionState | None = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def load_analysis_records() -> list[dict]:
    return RuntimeStore(RUNTIME_DB_PATH).load_json_records(EMOTION_ANALYSIS_NAMESPACE)


def successful_emotion_snapshot(record: dict) -> dict[str, Any] | None:
    emotion = record.get("emotion")
    if record.get("success") is not True or not isinstance(emotion, str):
        return None
    timestamp = record.get("timestamp")
    turn_count = record.get("turn_count")
    if not isinstance(timestamp, str) or type(turn_count) is not int:
        return None
    emotion = emotion.strip()
    if not emotion:
        return None
    return {
        "emotion": emotion,
        "timestamp": timestamp,
        "turn_count": turn_count,
    }


def load_latest_successful_emotion() -> dict[str, Any] | None:
    records = load_analysis_records()
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        snapshot = successful_emotion_snapshot(record)
        if snapshot is None:
            continue
        return snapshot
    return None


def append_analysis_record(record: dict[str, Any]) -> None:
    with _ANALYSIS_LOCK:
        output_record = {"timestamp": _now_iso(), **record}
        RuntimeStore(RUNTIME_DB_PATH).append_json_record(
            EMOTION_ANALYSIS_NAMESPACE,
            output_record,
        )


def analyze_emotion(
    llm,
    records: list[dict],
    current_input: str,
    *,
    previous_emotion: str = "",
    likely_emotions: list[str] | None = None,
    turn_count: int,
    emotion_interval: int,
) -> EmotionAnalysisResult:
    """执行单次情感分析：构建 prompt → 调用 LLM → 解析结果 → 持久化记录。"""
    dialogue_context = _dialogue_context(records, current_input, emotion_interval)
    prompt = build_emotion_prompt(
        records,
        current_input,
        previous_emotion=previous_emotion,
        likely_emotions=likely_emotions,
        max_turns=emotion_interval,
    )
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        output = content if isinstance(content, str) else str(content)
        state = emotion_state_from_output(output)
        emotion = state.primary_emotion if state else parse_emotion_output(output)
        if emotion is None:
            append_analysis_record({
                "turn_count": turn_count,
                "emotion_interval": emotion_interval,
                "input": prompt,
                "dialogue_context": dialogue_context,
                "output": output,
                "emotion": "",
                "state": {},
                "success": False,
                "error": "Failed to parse a known emotion label.",
            })
            return EmotionAnalysisResult("", prompt, output, False, "Failed to parse a known emotion label.")

        if state is None:
            state = EmotionState(primary_emotion=emotion)
        append_analysis_record({
            "turn_count": turn_count,
            "emotion_interval": emotion_interval,
            "input": prompt,
            "dialogue_context": dialogue_context,
            "output": output,
            "emotion": emotion,
            "state": state.to_dict(),
            "success": True,
            "error": "",
        })
        return EmotionAnalysisResult(emotion, prompt, output, True, state=state)
    except Exception as exc:
        append_analysis_record({
            "turn_count": turn_count,
            "emotion_interval": emotion_interval,
            "input": prompt,
            "dialogue_context": dialogue_context,
            "output": "",
            "emotion": "",
            "state": {},
            "success": False,
            "error": str(exc),
        })
        return EmotionAnalysisResult("", prompt, "", False, str(exc))
