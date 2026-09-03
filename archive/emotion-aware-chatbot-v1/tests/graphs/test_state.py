import pytest

from chatbot.graphs.state import capped_timeline
from chatbot.models.graph import ThreadRecord


def test_capped_timeline_keeps_newest_items():
    """Catches emotion timeline reducers retaining stale turns instead of the newest limit."""
    assert capped_timeline(
        [{"turn_count": 1}, {"turn_count": 2}],
        [{"turn_count": 3}],
        limit=2,
    ) == [{"turn_count": 2}, {"turn_count": 3}]


@pytest.mark.parametrize("timestamp", ["2026-09-01T08:00:00", "2026-09-01T08:00:00+08:00"])
def test_thread_record_rejects_non_utc_or_naive_timestamp_strings(timestamp):
    """Catches durable thread records storing ambiguous or non-UTC timestamp strings."""
    with pytest.raises(ValueError, match="UTC ISO-8601"):
        ThreadRecord(
            thread_id="thread-1",
            title="Test thread",
            created_at=timestamp,
            updated_at="2026-09-01T00:00:00+00:00",
        )
