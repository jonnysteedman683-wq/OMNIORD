"""Tests for the live progress tracker."""

from __future__ import annotations

from omniord.core.events import Event, EventType
from omniord.progress import ProgressTracker


def _event(etype: EventType, **payload: object) -> Event:
    return Event(type=etype.value, payload=payload)


def test_tracker_follows_node_lifecycle() -> None:
    tracker = ProgressTracker()
    tracker.handle(_event(EventType.DAG_STARTED, total=3))
    tracker.handle(_event(EventType.NODE_RUNNING, id="a"))
    tracker.handle(_event(EventType.NODE_COMPLETED, id="a"))
    tracker.handle(_event(EventType.NODE_RUNNING, id="b"))
    tracker.handle(_event(EventType.NODE_FAILED, id="b"))
    tracker.handle(_event(EventType.NODE_SKIPPED, id="c"))
    tracker.handle(_event(EventType.DAG_COMPLETED))

    assert tracker.total == 3
    assert tracker.done is True
    assert tracker.status == {"a": "completed", "b": "failed", "c": "skipped"}
    assert tracker.counts["completed"] == 1


def test_tracker_records_retries() -> None:
    tracker = ProgressTracker()
    tracker.handle(_event(EventType.NODE_RUNNING, id="a"))
    tracker.handle(_event(EventType.NODE_RETRY, id="a", attempt=2, error="boom"))
    assert tracker.status["a"] == "retrying"
    assert tracker.attempts["a"] == 2


def test_tracker_render_returns_table() -> None:
    from rich.table import Table

    tracker = ProgressTracker()
    tracker.handle(_event(EventType.DAG_STARTED, total=1))
    tracker.handle(_event(EventType.NODE_COMPLETED, id="a"))
    assert isinstance(tracker.render(), Table)
