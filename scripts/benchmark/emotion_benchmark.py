from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from chatbot.emotion_labels import EMOTION_LABEL_SET


LANGUAGES = {"zh", "en"}
SUBSETS = {
    "empathetic_dialogues_test",
    "empathetic_dialogues_balanced_seed",
    "empathetic_dialogues_context_diagnostic",
}
ANNOTATION_STATUSES = {"candidate", "annotated", "adjudicated", "released", "rejected"}
AMBIGUITY_LEVELS = {"low", "medium", "high"}
CONTEXT_DEPENDENCIES = {"none", "low", "medium", "high"}
SOURCE_STAGES = {"raw", "annotation", "release"}
LABEL_PROVENANCES = {
    "human_annotation",
    "human_authored_emotion_grounding",
}
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
SCENARIO_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


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
    _validate_scenario(record, errors)
    _validate_enum(record, "annotation_status", ANNOTATION_STATUSES, errors)
    if record.get("subset") in SUBSETS:
        _validate_enum(record, "label_provenance", LABEL_PROVENANCES, errors)

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
        if (
            record.get("source_stage") == "release"
            and record.get("annotation_status") not in {"adjudicated", "released"}
        ):
            errors.append("release packaging records must use adjudicated or released status")
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
    exported = {"id": record["case_id"], "expected": record["expected"]}
    if "label_provenance" in record:
        exported["label_provenance"] = record["label_provenance"]
    return exported


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


def _validate_scenario(record: dict[str, Any], errors: list[str]) -> None:
    scenario = _clean_string(record.get("scenario"))
    if not scenario:
        errors.append("scenario must be a non-empty string")
    elif SCENARIO_PATTERN.fullmatch(scenario) is None:
        errors.append("scenario must be lowercase snake case")


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
