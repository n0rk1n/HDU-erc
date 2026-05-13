import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = "chat_history.json"


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
