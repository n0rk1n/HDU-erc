import json
from pathlib import Path

PROFILE_FILE = "user_profile.json"


def load_profile() -> dict[str, str]:
    try:
        path = Path(PROFILE_FILE)
        if not path.exists() or path.stat().st_size == 0:
            return {}
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def format_profile(profile: dict[str, str]) -> str:
    if not profile:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in profile.items())
