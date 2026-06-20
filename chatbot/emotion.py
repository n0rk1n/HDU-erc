"""情感分析模块 —— 在聊天过程中按固定轮次间隔分析用户当前情绪，结果持久化到 JSON 文件。"""

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from chatbot.emotion_examples import DEFAULT_EMOTION_EXAMPLES
from chatbot.emotion_labels import EMOTION_LABELS, EMOTION_LABEL_SET
from chatbot.emotion_prompt import build_emotion_analysis_prompt
from chatbot.emotion_retrieval import select_dynamic_examples
from chatbot.emotion_state import EmotionState, emotion_state_from_output

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_EMOTION_ANALYSIS_FILE = str(DATA_DIR / "records" / "emotion_analysis.json")
DEFAULT_LEGACY_EMOTION_ANALYSIS_FILE = str(DATA_DIR / "emotion_analysis.json")
EMOTION_ANALYSIS_FILE = DEFAULT_EMOTION_ANALYSIS_FILE
LEGACY_EMOTION_ANALYSIS_FILE = DEFAULT_LEGACY_EMOTION_ANALYSIS_FILE
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


def build_emotion_prompt(
    records: list[dict],
    current_input: str,
    *,
    previous_emotion: str = "",
    likely_emotions: list[str] | None = None,
    max_turns: int = 5,
) -> str:
    utterances = _recent_contents(records, max_turns)
    current_input = current_input.strip()
    if current_input:
        utterances.append(current_input)
    dialogue_context = "</s>".join(utterances)
    retrieval_likely_emotions = [
        emotion
        for emotion in [previous_emotion, *(likely_emotions or [])]
        if emotion
    ]
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
        previous_emotion=previous_emotion,
        likely_emotions=likely_emotions,
        examples=selected_examples,
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
    primary_path = Path(EMOTION_ANALYSIS_FILE)
    if primary_path.exists():
        return _load_analysis_file(primary_path)

    if _should_read_legacy_analysis():
        return _load_analysis_file(Path(LEGACY_EMOTION_ANALYSIS_FILE))
    return []


def _should_read_legacy_analysis() -> bool:
    return (
        Path(EMOTION_ANALYSIS_FILE) == Path(DEFAULT_EMOTION_ANALYSIS_FILE)
        or Path(LEGACY_EMOTION_ANALYSIS_FILE) != Path(DEFAULT_LEGACY_EMOTION_ANALYSIS_FILE)
    )


def _load_analysis_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return []
        data = json.loads(content)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return data


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
        path = Path(EMOTION_ANALYSIS_FILE)
        records = load_analysis_records()
        output_record = {"timestamp": _now_iso(), **record}
        records.append(output_record)
        tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            print(f"Warning: failed to write emotion analysis: {exc}")
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


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
            "output": "",
            "emotion": "",
            "state": {},
            "success": False,
            "error": str(exc),
        })
        return EmotionAnalysisResult("", prompt, "", False, str(exc))
