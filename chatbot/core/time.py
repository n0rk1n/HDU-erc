from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 text form for SQLite."""
    return datetime.now(UTC).isoformat()


utc_now_iso8601 = utc_now
