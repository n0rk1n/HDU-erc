"""聊天历史持久化 —— 将对话记录读写到本地 SQLite，并提供格式化输出。"""

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from chatbot.runtime_store import DEFAULT_RUNTIME_DB_PATH, RuntimeStore

RUNTIME_DB_PATH = DEFAULT_RUNTIME_DB_PATH
HISTORY_NAMESPACE = "chat_history"
_HISTORY_LOCK = threading.RLock()

REGENERATION_REASONS = {
    "不准确",
    "不完整",
    "没有理解我的问题",
    "语气不合适",
    "其他",
}


@dataclass(frozen=True)
class FeedbackUpdateResult:
    status: str
    feedback: str = ""


@dataclass(frozen=True)
class RegenerationUpdateResult:
    status: str
    original_message_id: str = ""
    message_id: str = ""
    content: str = ""
    reason: str = ""
    original_user_message: str = ""


def load_history() -> list[dict]:
    return RuntimeStore(RUNTIME_DB_PATH).load_json_records(HISTORY_NAMESPACE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save_history(records: list[dict]) -> bool:
    return RuntimeStore(RUNTIME_DB_PATH).replace_json_records(HISTORY_NAMESPACE, records)


def append_message(role: str, content: str) -> dict | None:
    """追加一条消息到本地 SQLite 历史记录。"""
    with _HISTORY_LOCK:
        record = {
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
        }
        if RuntimeStore(RUNTIME_DB_PATH).append_json_record(HISTORY_NAMESPACE, record):
            return record
        return None


def append_ai_message(content: str, **extra: Any) -> dict | None:
    with _HISTORY_LOCK:
        record = _new_ai_record(content, **extra)
        if RuntimeStore(RUNTIME_DB_PATH).append_json_record(HISTORY_NAMESPACE, record):
            return record
        return None


def _new_ai_record(content: str, **extra: Any) -> dict:
    record = {
        "id": f"ai_{uuid4().hex}",
        "role": "ai",
        "content": content,
        "timestamp": _now_iso(),
        "feedback": None,
    }
    record.update(extra)
    return record


def _nearest_preceding_human_content(records: list[dict], index: int) -> str:
    for record in reversed(records[:index]):
        if record.get("role") == "human":
            content = str(record.get("content", "")).strip()
            if content:
                return content
    return ""


def _resolve_original_user_message(records: list[dict], index: int) -> str:
    record = records[index]
    seen_ids: set[str] = set()
    while True:
        regenerated_from = record.get("regenerated_from")
        if not regenerated_from:
            return _nearest_preceding_human_content(records, index)

        parent_id = str(regenerated_from)
        if parent_id in seen_ids:
            return ""
        seen_ids.add(parent_id)

        parent_index = -1
        parent_record: dict | None = None
        for candidate_index, candidate in enumerate(records):
            if candidate.get("id") == parent_id:
                parent_index = candidate_index
                parent_record = candidate
                break
        if parent_record is None:
            return ""

        parent_regeneration = parent_record.get("regeneration")
        if isinstance(parent_regeneration, dict):
            original_user_message = str(
                parent_regeneration.get("original_user_message", "")
            ).strip()
            if original_user_message:
                return original_user_message

        record = parent_record
        index = parent_index


def record_message_feedback(message_id: str, feedback: str) -> FeedbackUpdateResult:
    if feedback not in {"like", "dislike"}:
        return FeedbackUpdateResult("invalid_feedback")

    with _HISTORY_LOCK:
        records = load_history()
        for record in records:
            if record.get("id") != message_id:
                continue
            if record.get("role") != "ai":
                return FeedbackUpdateResult("not_ai")
            existing_feedback = record.get("feedback")
            if existing_feedback in {"like", "dislike"}:
                return FeedbackUpdateResult("already_rated", existing_feedback)

            record["feedback"] = feedback
            if _save_history(records):
                return FeedbackUpdateResult("updated", feedback)
            return FeedbackUpdateResult("write_failed")

    return FeedbackUpdateResult("not_found")


def prepare_message_regeneration(
    message_id: str,
    reason: str,
) -> RegenerationUpdateResult:
    if reason not in REGENERATION_REASONS:
        return RegenerationUpdateResult("invalid_reason")

    with _HISTORY_LOCK:
        records = load_history()
        for index, record in enumerate(records):
            if record.get("id") != message_id:
                continue
            if record.get("role") != "ai":
                return RegenerationUpdateResult("not_ai")
            existing_regeneration = record.get("regeneration")
            if isinstance(existing_regeneration, dict):
                return RegenerationUpdateResult(
                    "already_regenerated",
                    original_message_id=message_id,
                    message_id=str(existing_regeneration.get("regenerated_message_id", "")),
                    reason=str(existing_regeneration.get("reason", "")),
                    original_user_message=str(
                        existing_regeneration.get("original_user_message", "")
                    ),
                )
            original_user_message = _resolve_original_user_message(records, index)
            if not original_user_message:
                return RegenerationUpdateResult(
                    "missing_prompt",
                    original_message_id=message_id,
                )
            return RegenerationUpdateResult(
                "ready",
                original_message_id=message_id,
                reason=reason,
                original_user_message=original_user_message,
            )

    return RegenerationUpdateResult("not_found")


def record_message_regeneration(
    message_id: str,
    reason: str,
    new_content: str,
) -> RegenerationUpdateResult:
    if reason not in REGENERATION_REASONS:
        return RegenerationUpdateResult("invalid_reason")

    with _HISTORY_LOCK:
        records = load_history()
        for index, record in enumerate(records):
            if record.get("id") != message_id:
                continue
            if record.get("role") != "ai":
                return RegenerationUpdateResult("not_ai")
            existing_regeneration = record.get("regeneration")
            if isinstance(existing_regeneration, dict):
                return RegenerationUpdateResult(
                    "already_regenerated",
                    original_message_id=message_id,
                    message_id=str(existing_regeneration.get("regenerated_message_id", "")),
                    reason=str(existing_regeneration.get("reason", "")),
                    original_user_message=str(
                        existing_regeneration.get("original_user_message", "")
                    ),
                )

            original_user_message = _resolve_original_user_message(records, index)
            if not original_user_message:
                return RegenerationUpdateResult(
                    "missing_prompt",
                    original_message_id=message_id,
                )

            new_record = _new_ai_record(new_content, regenerated_from=message_id)
            record["regeneration"] = {
                "reason": reason,
                "regenerated_message_id": new_record["id"],
                "timestamp": _now_iso(),
                "original_user_message": original_user_message,
                "original_ai_content": str(record.get("content", "")),
            }
            records.append(new_record)
            if _save_history(records):
                return RegenerationUpdateResult(
                    "updated",
                    original_message_id=message_id,
                    message_id=new_record["id"],
                    content=new_content,
                    reason=reason,
                    original_user_message=original_user_message,
                )
            return RegenerationUpdateResult("write_failed", original_message_id=message_id)

    return RegenerationUpdateResult("not_found")


def format_recent(records: list[dict], n: int = 10) -> str:
    if not records:
        return ""
    if not isinstance(n, int) or n <= 0:
        n = 10
    recent = records[-n:]
    lines = []
    for record in recent:
        role = record.get("role")
        content = record.get("content", "")
        if role == "human":
            lines.append(f"You: {content}")
        elif role == "ai":
            lines.append(f"Bot: {content}")
    return "\n".join(lines)
