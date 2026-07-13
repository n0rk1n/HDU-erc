"""Invoke Codex CLI for structured emotion classification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.emotion import EMOTION_LABEL_SET, build_emotion_prompt
from scripts.run_emotion_ablation import (
    RUN_CONFIGS,
    _history_records,
    _likely_emotions,
    _previous_emotion,
    _turn_count,
    load_dialogues,
)


@dataclass(frozen=True)
class CodexResult:
    emotion: str
    output: str
    success: bool
    error: str = ""


def build_command(schema_file: Path, *, model: str | None) -> list[str]:
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--output-schema",
        str(schema_file),
    ]
    if model:
        command.extend(["--model", model])
    command.append("-")
    return command


def parse_result(output: str) -> CodexResult:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return CodexResult("", output, False, f"Invalid JSON: {exc}")
    emotion = payload.get("emotion") if isinstance(payload, dict) else None
    if emotion not in EMOTION_LABEL_SET:
        return CodexResult("", output, False, f"Unsupported emotion: {emotion!r}")
    return CodexResult(emotion, output, True)


def invoke_codex(
    prompt: str,
    *,
    schema_file: Path,
    model: str | None,
    timeout: int,
    run_subprocess: Any = subprocess.run,
) -> CodexResult:
    instruction = (
        "Classify the emotion using only the supplied prompt. "
        "Return exactly the JSON object required by the output schema.\n\n" + prompt
    )
    try:
        completed = run_subprocess(
            build_command(schema_file, model=model),
            input=instruction,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return CodexResult("", "", False, str(exc))
    if completed.returncode != 0:
        return CodexResult("", completed.stdout, False, completed.stderr.strip())
    return parse_result(completed.stdout.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emotion-recognition ablations with Codex CLI."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
