"""用户画像加载 —— 从本地 SQLite 读取 key-value 用户信息，注入系统提示词。"""

from chatbot.runtime_store import DEFAULT_RUNTIME_DB_PATH, RuntimeStore

RUNTIME_DB_PATH = DEFAULT_RUNTIME_DB_PATH


def load_profile() -> dict[str, str]:
    return RuntimeStore(RUNTIME_DB_PATH).load_profile()


def save_profile(profile: dict[str, str]) -> bool:
    return RuntimeStore(RUNTIME_DB_PATH).replace_profile(profile)


def format_profile(profile: dict[str, str]) -> str:
    if not profile:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in profile.items())
