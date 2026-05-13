import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = "chat_history.json"


def load_history() -> list[dict]:
    path = Path(HISTORY_FILE)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_message(role: str, content: str) -> None:
    try:
        records = load_history()
        records.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        path = Path(HISTORY_FILE)
        with path.open("w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"Warning: could not save chat history: {exc}")
