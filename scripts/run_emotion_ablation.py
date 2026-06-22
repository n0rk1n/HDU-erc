"""Run first-stage emotion-recognition ablation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.config import load_config
from chatbot.emotion import build_emotion_prompt, parse_emotion_output
from chatbot.emotion_state import EmotionState, emotion_state_from_output
from chatbot.llm_adapter import build_chat_model


@dataclass(frozen=True)
class AblationRunConfig:
    name: str
    example_mode: str
    include_emotion_history: bool
    max_turns: int | None = None


RUN_CONFIGS = {
    "full": AblationRunConfig("full", "dynamic", True),
    "no_dynamic_examples": AblationRunConfig("no_dynamic_examples", "static", True),
    "no_emotion_history": AblationRunConfig("no_emotion_history", "dynamic", False),
    "short_context": AblationRunConfig("short_context", "dynamic", True, max_turns=1),
    "zero_shot": AblationRunConfig("zero_shot", "none", False),
}


def load_dialogues(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Invalid dialogue record at {path}:{line_number}: expected object.")
        if not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ValueError(f"Invalid dialogue record at {path}:{line_number}: missing id.")
        if not isinstance(item.get("current_input"), str) or not item["current_input"].strip():
            raise ValueError(f"Invalid dialogue record at {path}:{line_number}: missing current_input.")
        history = item.get("history", [])
        if not isinstance(history, list):
            raise ValueError(f"Invalid dialogue record at {path}:{line_number}: history must be a list.")
        records.append(item)
    return records


def _history_records(case: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in case.get("history", []):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"human", "ai"} and isinstance(content, str):
            records.append({"role": role, "content": content})
    return records


def _turn_count(case: dict[str, Any], fallback: int) -> int:
    value = case.get("turn_count")
    return value if type(value) is int and value > 0 else fallback


def _previous_emotion(records: list[dict[str, Any]]) -> str:
    for record in reversed(records):
        emotion = record.get("emotion") or record.get("predicted_emotion")
        if isinstance(emotion, str) and emotion.strip():
            return emotion.strip()
    return ""


def _likely_emotions(records: list[dict[str, Any]]) -> list[str]:
    emotions = []
    for record in reversed(records):
        emotion = record.get("emotion") or record.get("predicted_emotion")
        if isinstance(emotion, str):
            cleaned = emotion.strip()
            if cleaned and cleaned not in emotions:
                emotions.append(cleaned)
        if len(emotions) >= 3:
            break
    return emotions


def _content(response: Any) -> str:
    content = response.content if hasattr(response, "content") else response
    return content if isinstance(content, str) else str(content)


def _state_and_emotion(output: str) -> tuple[EmotionState | None, str]:
    state = emotion_state_from_output(output)
    if state is not None:
        return state, state.primary_emotion
    emotion = parse_emotion_output(output) or ""
    if emotion:
        return EmotionState(primary_emotion=emotion), emotion
    return None, ""


def run_config(
    config: AblationRunConfig,
    dialogues_file: Path,
    output_file: Path,
    llm: Any,
    *,
    emotion_interval: int,
) -> int:
    cases = load_dialogues(dialogues_file)
    records = []
    for index, case in enumerate(cases, start=1):
        history = _history_records(case)
        turn_count = _turn_count(case, index)
        max_turns = config.max_turns or emotion_interval
        prompt = build_emotion_prompt(
            history,
            case["current_input"],
            previous_emotion=_previous_emotion(history),
            likely_emotions=_likely_emotions(history),
            max_turns=max_turns,
            example_mode=config.example_mode,
            include_emotion_history=config.include_emotion_history,
        )
        try:
            output = _content(llm.invoke(prompt))
            state, emotion = _state_and_emotion(output)
            success = bool(emotion)
            error = "" if success else "Failed to parse a known emotion label."
        except Exception as exc:
            output = ""
            state = None
            emotion = ""
            success = False
            error = str(exc)
        records.append({
            "case_id": case["id"].strip(),
            "run": config.name,
            "turn_count": turn_count,
            "emotion_interval": max_turns,
            "input": prompt,
            "output": output,
            "emotion": emotion,
            "state": state.to_dict() if state is not None else {},
            "success": success,
            "error": error,
        })
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run emotion-recognition ablation experiments.")
    parser.add_argument("--dialogues-file", default="data/examples/ablation_dialogues.jsonl")
    parser.add_argument("--output-dir", default="data/records/ablation")
    parser.add_argument(
        "--run",
        action="append",
        choices=sorted(RUN_CONFIGS),
        help="Run name to execute. Defaults to all runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chat_config = load_config([])
    llm = build_chat_model(chat_config.emotion_llm)
    run_names = args.run or list(RUN_CONFIGS)
    for run_name in run_names:
        output_file = Path(args.output_dir) / f"{run_name}.json"
        count = run_config(
            RUN_CONFIGS[run_name],
            Path(args.dialogues_file),
            output_file,
            llm,
            emotion_interval=chat_config.emotion_interval,
        )
        print(f"{run_name}: wrote {count} records to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
