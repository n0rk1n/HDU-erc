"""聊天历史持久化 —— 将对话记录读写到 JSON 文件，并提供格式化输出。"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = "../data/chat_history.json"


def load_history() -> list[dict]:
    try:
        path = Path(HISTORY_FILE)
        if not path.exists() or path.stat().st_size == 0:
            return []
        with path.open() as f:
            records = json.load(f)
        if isinstance(records, list):
            return records
        return []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def append_message(role: str, content: str) -> None:
    """追加一条消息到历史文件 —— 先写临时文件再原子替换，避免写坏 JSON。"""
    try:
        records = load_history()
        records.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        path = Path(HISTORY_FILE)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            tmp.replace(path)
        except OSError as exc:
            print(f"Warning: could not save chat history: {exc}")
        finally:
            if tmp.exists():
                tmp.unlink()
    except OSError as exc:
        print(f"Warning: could not save chat history: {exc}")


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
