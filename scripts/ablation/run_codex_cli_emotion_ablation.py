"""Invoke Codex CLI for structured emotion classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chatbot.emotion import EMOTION_LABEL_SET, build_emotion_prompt
from scripts.ablation.run_emotion_ablation import (
    RUN_CONFIGS,
    AblationRunConfig,
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


RUNTIME_PROVENANCE = "codex_exec_ephemeral_read_only_isolated_cwd_v1"


def build_command(schema_file: Path, *, model: str | None) -> list[str]:
    schema_file = schema_file.resolve()
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
        with tempfile.TemporaryDirectory(prefix="codex-emotion-") as isolated_cwd:
            repository_root = Path(__file__).resolve().parents[2]
            isolated_path = Path(isolated_cwd).resolve()
            if isolated_path == repository_root or repository_root in isolated_path.parents:
                raise RuntimeError("Temporary Codex working directory is inside the repository")
            completed = run_subprocess(
                build_command(schema_file.resolve(), model=model),
                input=instruction,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                cwd=isolated_cwd,
            )
    except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
        return CodexResult("", "", False, str(exc))
    if completed.returncode != 0:
        return CodexResult("", completed.stdout, False, completed.stderr.strip())
    return parse_result(completed.stdout.strip())


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Invalid existing result file: {path}")
    return payload


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_from_result(
    case: dict[str, Any],
    config: AblationRunConfig,
    prompt: str,
    result: CodexResult,
    index: int,
    *,
    emotion_interval: int,
    provenance: dict[str, str],
) -> dict[str, Any]:
    return {
        "case_id": case["id"].strip(),
        "run": config.name,
        "turn_count": _turn_count(case, index),
        "emotion_interval": emotion_interval,
        "input": prompt,
        "output": result.output,
        "emotion": result.emotion,
        "success": result.success,
        "error": result.error,
        **provenance,
    }


def build_case_prompt(
    config: AblationRunConfig,
    case: dict[str, Any],
    index: int,
    *,
    emotion_interval: int,
) -> str:
    history = _history_records(case)
    return build_emotion_prompt(
        history,
        case["current_input"],
        previous_emotion=_previous_emotion(history),
        likely_emotions=_likely_emotions(history),
        max_turns=config.max_turns or emotion_interval,
        example_mode=config.example_mode,
        include_emotion_history=config.include_emotion_history,
    )


def noop_runs_against_full(
    cases: list[dict[str, Any]],
    run_names: list[str],
    *,
    emotion_interval: int,
) -> list[str]:
    """Detect treatments whose prompts are identical to full for every case."""
    full = RUN_CONFIGS["full"]
    full_prompts = [
        build_case_prompt(full, case, index, emotion_interval=emotion_interval)
        for index, case in enumerate(cases, start=1)
    ]
    return [
        run_name
        for run_name in run_names
        if run_name != "full" and all(
            build_case_prompt(
                RUN_CONFIGS[run_name], case, index, emotion_interval=emotion_interval
            )
            == full_prompt
            for index, (case, full_prompt) in enumerate(
                zip(cases, full_prompts), start=1
            )
        )
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_resume_provenance(
    prompt: str,
    *,
    schema_file: Path,
    model: str | None,
    codex_cli_version: str,
) -> dict[str, str]:
    schema_bytes = schema_file.read_bytes() if schema_file.exists() else b"<missing-schema>"
    return {
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "model": model or "<codex-default>",
        "schema_sha256": _sha256_bytes(schema_bytes),
        "codex_cli_version": codex_cli_version,
        "runtime_provenance": RUNTIME_PROVENANCE,
    }


def detect_codex_cli_version(run_subprocess: Any = subprocess.run) -> str:
    completed = run_subprocess(
        ["codex", "--version"], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"Unable to determine Codex CLI version: {completed.stderr.strip()}")
    return completed.stdout.strip()


def run_ablation(
    config: AblationRunConfig,
    cases: list[dict[str, Any]],
    output_file: Path,
    *,
    schema_file: Path,
    model: str | None,
    timeout: int,
    retries: int,
    emotion_interval: int,
    codex_cli_version: str = "unknown",
    invoke: Any = invoke_codex,
) -> list[dict[str, Any]]:
    if retries not in {0, 1}:
        raise ValueError("retries must be 0 or 1")
    existing = _load_existing(output_file)
    existing_by_case = {
        record["case_id"]: record
        for record in existing
        if (
            isinstance(record.get("case_id"), str)
            and record.get("run") == config.name
            and record.get("success") is True
        )
    }
    by_case: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, start=1):
        case_id = case["id"].strip()
        prompt = build_case_prompt(
            config, case, index, emotion_interval=emotion_interval
        )
        provenance = build_resume_provenance(
            prompt,
            schema_file=schema_file,
            model=model,
            codex_cli_version=codex_cli_version,
        )
        existing_record = existing_by_case.get(case_id)
        if existing_record is not None and all(
            existing_record.get(key) == value for key, value in provenance.items()
        ):
            by_case[case_id] = existing_record
            continue
        result = CodexResult("", "", False, "not invoked")
        for _ in range(retries + 1):
            result = invoke(
                prompt,
                schema_file=schema_file,
                model=model,
                timeout=timeout,
            )
            if result.success:
                break
        by_case[case_id] = _record_from_result(
            case,
            config,
            prompt,
            result,
            index,
            emotion_interval=config.max_turns or emotion_interval,
            provenance=provenance,
        )
        ordered_records = [
            by_case[item["id"].strip()]
            for item in cases
            if item["id"].strip() in by_case
        ]
        _write_records(output_file, ordered_records)
    return [by_case[item["id"].strip()] for item in cases]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emotion-recognition ablations with Codex CLI."
    )
    parser.add_argument("--dialogues-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--schema-file",
        default="data/config/codex_emotion_result.schema.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--run",
        action="append",
        choices=sorted(RUN_CONFIGS),
        help="Run name to execute. Defaults to all runs.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--codex-version", help="Override detected Codex CLI version.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--emotion-interval", type=int, default=5)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries not in {0, 1}:
        parser.error("--retries must be 0 or 1")
    if args.emotion_interval <= 0:
        parser.error("--emotion-interval must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_dialogues(Path(args.dialogues_file))
    if args.limit is not None:
        cases = cases[:args.limit]
    run_names = args.run or list(RUN_CONFIGS)
    noops = noop_runs_against_full(
        cases, run_names, emotion_interval=args.emotion_interval
    )
    for run_name in noops:
        print(
            f"{run_name}: skipped because every prompt is identical to full "
            "for this dataset"
        )
    run_names = [run_name for run_name in run_names if run_name not in noops]
    if not run_names:
        return 0
    codex_cli_version = args.codex_version or detect_codex_cli_version()
    for run_name in run_names:
        output_file = Path(args.output_dir) / f"{run_name}.json"
        records = run_ablation(
            RUN_CONFIGS[run_name],
            cases,
            output_file,
            schema_file=Path(args.schema_file),
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            emotion_interval=args.emotion_interval,
            codex_cli_version=codex_cli_version,
        )
        print(f"{run_name}: wrote {len(records)} records to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
