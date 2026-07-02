"""Shared helpers for the emotion ablation v2 benchmark."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from chatbot.emotion_labels import EMOTION_LABEL_SET


BENCHMARK_ROOT = Path("data/benchmarks/emotion_ablation_v2")

LANGUAGES = {"zh", "en"}
SUBSETS = {"core_parallel", "extended_independent", "challenge", "seed", "legacy_compat"}
SEED_GROUPS = {"core_parallel_seed", "independent_seed"}
ANNOTATION_STATUSES = {"candidate", "annotated", "adjudicated", "released", "rejected"}
AMBIGUITY_LEVELS = {"low", "medium", "high"}
CONTEXT_DEPENDENCIES = {"none", "low", "medium", "high"}
CONTEXT_DEPENDENCY_LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}
SOURCE_STAGES = {"raw", "annotation", "release"}
HISTORY_ROLES = {"human", "ai"}
QUALITY_FLAGS = {
    "too_template_like",
    "emotion_too_explicit",
    "emotion_evidence_weak",
    "parallel_mismatch",
    "label_boundary_case",
    "safety_sensitive",
    "requires_context",
    "contains_irony",
    "mixed_emotion",
    "cultural_specificity",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected object")
        records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require_string(record, "case_id", errors)
    _validate_enum(record, "language", LANGUAGES, errors)
    _validate_enum(record, "subset", SUBSETS, errors)
    _validate_emotion(record.get("expected"), "expected", errors)
    _validate_turn_count(record, errors)
    _validate_history(record, errors)
    _require_string(record, "current_input", errors)
    _require_string(record, "scenario", errors)
    _validate_enum(record, "annotation_status", ANNOTATION_STATUSES, errors)

    if record.get("subset") == "core_parallel" and not _clean_string(record.get("pair_id")):
        errors.append("core_parallel records must include pair_id")
    if record.get("subset") == "seed" and "seed_group" in record:
        _validate_enum(record, "seed_group", SEED_GROUPS, errors)
    if "target_emotion" in record:
        _validate_emotion(record.get("target_emotion"), "target_emotion", errors)
    if "secondary_emotions" in record:
        _validate_secondary_emotions(record.get("secondary_emotions"), errors)
    if "intensity" in record:
        _validate_intensity(record.get("intensity"), errors)
    if "ambiguity_level" in record:
        _validate_enum(record, "ambiguity_level", AMBIGUITY_LEVELS, errors)
    if "context_dependency" in record:
        _validate_enum(record, "context_dependency", CONTEXT_DEPENDENCIES, errors)
    if "source_stage" in record:
        _validate_enum(record, "source_stage", SOURCE_STAGES, errors)
    if "quality_flags" in record:
        _validate_quality_flags(record.get("quality_flags"), errors)
    return errors


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_case_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        case_id = _clean_string(record.get("case_id"))
        if case_id:
            if case_id in seen_case_ids:
                errors.append(f"duplicate case_id: {case_id}")
            seen_case_ids.add(case_id)
        for error in validate_record(record):
            marker = case_id or f"record {index}"
            errors.append(f"{marker}: {error}")
    return errors


def export_dialogue(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["case_id"],
        "turn_count": record["turn_count"],
        "history": record["history"],
        "current_input": record["current_input"],
        "notes": record.get("rationale", ""),
    }


def export_label(record: dict[str, Any]) -> dict[str, Any]:
    return {"id": record["case_id"], "expected": record["expected"]}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    return {
        "label": Counter(str(record.get("expected", "")) for record in records),
        "language": Counter(str(record.get("language", "")) for record in records),
        "subset": Counter(str(record.get("subset", "")) for record in records),
        "scenario": Counter(str(record.get("scenario", "")) for record in records),
        "ambiguity_level": Counter(str(record.get("ambiguity_level", "")) for record in records),
        "context_dependency": Counter(str(record.get("context_dependency", "")) for record in records),
        "quality_flags": Counter(
            flag
            for record in records
            for flag in record.get("quality_flags", [])
            if isinstance(flag, str)
        ),
    }


def parallel_equivalence_errors(
    records: list[dict[str, Any]], *, max_intensity_delta: float = 0.15
) -> list[str]:
    errors: list[str] = []
    by_pair_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("subset") not in {"core_parallel", "seed"}:
            continue
        pair_id = _clean_string(record.get("pair_id"))
        if pair_id:
            by_pair_id[pair_id].append(record)

    for pair_id, pair_records in sorted(by_pair_id.items()):
        languages = {record.get("language") for record in pair_records}
        if languages != {"zh", "en"}:
            errors.append(f"{pair_id}: expected one zh and one en record")
            continue
        if len(pair_records) != 2:
            errors.append(f"{pair_id}: expected exactly 2 records")
            continue
        first, second = pair_records
        if first.get("expected") != second.get("expected"):
            errors.append(f"{pair_id}: expected labels differ")
        if _history_roles(first) != _history_roles(second):
            errors.append(f"{pair_id}: history role sequence differs")
        first_intensity = first.get("intensity")
        second_intensity = second.get("intensity")
        if isinstance(first_intensity, (int, float)) and isinstance(second_intensity, (int, float)):
            if abs(float(first_intensity) - float(second_intensity)) > max_intensity_delta:
                errors.append(f"{pair_id}: intensity delta exceeds {max_intensity_delta}")
        first_context_dependency = first.get("context_dependency")
        second_context_dependency = second.get("context_dependency")
        if (
            first_context_dependency in CONTEXT_DEPENDENCY_LEVELS
            and second_context_dependency in CONTEXT_DEPENDENCY_LEVELS
            and abs(
                CONTEXT_DEPENDENCY_LEVELS[first_context_dependency]
                - CONTEXT_DEPENDENCY_LEVELS[second_context_dependency]
            )
            > 1
        ):
            errors.append(f"{pair_id}: context_dependency differs by more than one level")
    return errors


def _require_string(record: dict[str, Any], key: str, errors: list[str]) -> None:
    if not _clean_string(record.get(key)):
        errors.append(f"{key} must be a non-empty string")


def _validate_enum(record: dict[str, Any], key: str, allowed: set[str], errors: list[str]) -> None:
    value = record.get(key)
    if value not in allowed:
        errors.append(f"{key} must be one of: {', '.join(sorted(allowed))}")


def _validate_emotion(value: Any, key: str, errors: list[str]) -> None:
    if not isinstance(value, str) or value.strip().lower() not in EMOTION_LABEL_SET:
        errors.append(f"{key} must be one of the supported emotion labels")


def _validate_turn_count(record: dict[str, Any], errors: list[str]) -> None:
    if type(record.get("turn_count")) is not int or record["turn_count"] < 1:
        errors.append("turn_count must be a positive integer")


def _validate_history(record: dict[str, Any], errors: list[str]) -> None:
    history = record.get("history")
    if not isinstance(history, list):
        errors.append("history must be a list")
        return
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            errors.append(f"history[{index}] must be an object")
            continue
        if item.get("role") not in HISTORY_ROLES:
            errors.append(f"history[{index}].role must be human or ai")
        if not isinstance(item.get("content"), str):
            errors.append(f"history[{index}].content must be a string")


def _validate_secondary_emotions(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("secondary_emotions must be a list")
        return
    if len(value) > 3:
        errors.append("secondary_emotions must contain at most 3 labels")
    for emotion in value:
        _validate_emotion(emotion, "secondary_emotions", errors)


def _validate_intensity(value: Any, errors: list[str]) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        errors.append("intensity must be a number from 0.0 to 1.0")


def _validate_quality_flags(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("quality_flags must be a list")
        return
    for flag in value:
        if flag not in QUALITY_FLAGS:
            errors.append(f"unknown quality flag: {flag}")


def _clean_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _history_roles(record: dict[str, Any]) -> list[str]:
    return [
        item.get("role", "")
        for item in record.get("history", [])
        if isinstance(item, dict)
    ]
