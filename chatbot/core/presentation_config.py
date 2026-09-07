"""Display names for public emotion projections, independent of model labels."""
from __future__ import annotations

import os
from pathlib import Path

from chatbot.core.errors import ConfigError
from chatbot.core.paths import PROJECT_ROOT
from chatbot.emotion.config import read_json


def load_emotion_names() -> dict[str, str]:
    configured_path = os.getenv("EMOTION_NAMES_PATH", "").strip()
    path = Path(configured_path) if configured_path else PROJECT_ROOT / "data/config/emotion_names.json"
    names = read_json(path)
    if not isinstance(names, dict) or any(
        not label.strip() or not isinstance(name, str) or not name.strip()
        for label, name in names.items()
    ):
        raise ConfigError("emotion names must be an object mapping nonempty labels to nonempty strings")
    return names
