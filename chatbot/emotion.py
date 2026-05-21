"""情感分析模块 —— 在聊天过程中按固定轮次间隔分析用户当前情绪，结果持久化到 JSON 文件。"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_EMOTION_ANALYSIS_FILE = str(DATA_DIR / "records" / "emotion_analysis.json")
DEFAULT_LEGACY_EMOTION_ANALYSIS_FILE = str(DATA_DIR / "emotion_analysis.json")
EMOTION_ANALYSIS_FILE = DEFAULT_EMOTION_ANALYSIS_FILE
LEGACY_EMOTION_ANALYSIS_FILE = DEFAULT_LEGACY_EMOTION_ANALYSIS_FILE
EMOTION_LABELS = [
    "surprised",
    "excited",
    "annoyed",
    "proud",
    "angry",
    "sad",
    "grateful",
    "lonely",
    "impressed",
    "afraid",
    "disgusted",
    "confident",
    "terrified",
    "hopeful",
    "anxious",
    "disappointed",
    "joyful",
    "prepared",
    "guilty",
    "furious",
    "nostalgic",
    "jealous",
    "anticipating",
    "embarrassed",
    "content",
    "devastated",
    "sentimental",
    "caring",
    "trusting",
    "ashamed",
    "apprehensive",
    "faithful",
]

EMOTION_LABEL_SET = set(EMOTION_LABELS)


@dataclass(frozen=True)
class EmotionAnalysisResult:
    emotion: str
    input: str
    output: str
    success: bool
    error: str = ""


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
    max_turns: int = 5,
) -> str:
    utterances = _recent_contents(records, max_turns)
    current_input = current_input.strip()
    if current_input:
        utterances.append(current_input)
    dialogue_context = "</s>".join(utterances)
    labels = ", ".join(EMOTION_LABELS)
    likely_line = ""
    if previous_emotion:
        likely_line = f"\n- More likely emotion label: {previous_emotion}"

    return f"""Infer the user's current emotion from the dialogue context.
- Dialogue context: The conversation history between user and assistant, with utterances separated by </s>.
- Emotion labels: {labels}
- Choose a single inferred emotion from the provided Emotion labels, not outside of them.
- Response Format: Emotion: [a single inferred emotion]{likely_line}

Dialogue context: {dialogue_context}""".strip()


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
    path = Path(EMOTION_ANALYSIS_FILE)
    records = load_analysis_records()
    output_record = {"timestamp": _now_iso(), **record}
    records.append(output_record)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"Warning: failed to write emotion analysis: {exc}")


def analyze_emotion(
    llm,
    records: list[dict],
    current_input: str,
    *,
    previous_emotion: str = "",
    turn_count: int,
    emotion_interval: int,
) -> EmotionAnalysisResult:
    """执行单次情感分析：构建 prompt → 调用 LLM → 解析结果 → 持久化记录。"""
    prompt = build_emotion_prompt(
        records,
        current_input,
        previous_emotion=previous_emotion,
        max_turns=emotion_interval,
    )
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        output = content if isinstance(content, str) else str(content)
        emotion = parse_emotion_output(output)
        if emotion is None:
            append_analysis_record({
                "turn_count": turn_count,
                "emotion_interval": emotion_interval,
                "input": prompt,
                "output": output,
                "emotion": "",
                "success": False,
                "error": "Failed to parse a known emotion label.",
            })
            return EmotionAnalysisResult("", prompt, output, False, "Failed to parse a known emotion label.")

        append_analysis_record({
            "turn_count": turn_count,
            "emotion_interval": emotion_interval,
            "input": prompt,
            "output": output,
            "emotion": emotion,
            "success": True,
            "error": "",
        })
        return EmotionAnalysisResult(emotion, prompt, output, True)
    except Exception as exc:
        append_analysis_record({
            "turn_count": turn_count,
            "emotion_interval": emotion_interval,
            "input": prompt,
            "output": "",
            "emotion": "",
            "success": False,
            "error": str(exc),
        })
        return EmotionAnalysisResult("", prompt, "", False, str(exc))
