"""用户画像加载 —— 从 JSON 读取 key-value 用户信息，注入系统提示词以个性化对话。"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_FILE = str(PROJECT_ROOT / "data" / "config" / "user_profile.json")
DEFAULT_LEGACY_PROFILE_FILE = str(PROJECT_ROOT / "user_profile.json")
PROFILE_FILE = DEFAULT_PROFILE_FILE
LEGACY_PROFILE_FILE = DEFAULT_LEGACY_PROFILE_FILE


def load_profile() -> dict[str, str]:
    primary_path = Path(PROFILE_FILE)
    if primary_path.exists():
        return _load_profile_file(primary_path)

    if _should_read_legacy_profile():
        return _load_profile_file(Path(LEGACY_PROFILE_FILE))
    return {}


def _should_read_legacy_profile() -> bool:
    return (
        Path(PROFILE_FILE) == Path(DEFAULT_PROFILE_FILE)
        or Path(LEGACY_PROFILE_FILE) != Path(DEFAULT_LEGACY_PROFILE_FILE)
    )


def _load_profile_file(path: Path) -> dict[str, str]:
    try:
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
