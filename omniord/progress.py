"""Live progress tracking for a DAG run, driven by the event bus.

``ProgressTracker`` consumes :class:`~omniord.core.events.Event`s and maintains
the current status of each node — framework-agnostic and unit-testable. Its
:meth:`render` returns a Rich renderable so the CLI can show it inside a
``rich.live.Live`` display.
"""

from __future__ import annotations

from rich.table import Table

from omniord.core.events import Event, EventType

_STATUS_STYLE = {
    "running": ("running", "yellow"),
    "retrying": ("retrying", "magenta"),
    "completed": ("completed", "green"),
    "failed": ("failed", "red"),
    "skipped": ("skipped", "dim"),
}


class ProgressTracker:
    """Tracks per-node status from the stream of engine events."""

    def __init__(self) -> None:
        self.total = 0
        self.done = False
        self.status: dict[str, str] = {}
        self.attempts: dict[str, int] = {}

    def handle(self, event: Event) -> None:
        etype = event.type
        node_id = event.payload.get("id")
        if etype == EventType.DAG_STARTED.value:
            self.total = int(event.payload.get("total", 0))
        elif etype == EventType.DAG_COMPLETED.value:
            self.done = True
        elif etype == EventType.NODE_RUNNING.value and node_id:
            self.status[node_id] = "running"
        elif etype == EventType.NODE_COMPLETED.value and node_id:
            self.status[node_id] = "completed"
        elif etype == EventType.NODE_FAILED.value and node_id:
            self.status[node_id] = "failed"
        elif etype == EventType.NODE_SKIPPED.value and node_id:
            self.status[node_id] = "skipped"
        elif etype == EventType.NODE_RETRY.value and node_id:
            self.status[node_id] = "retrying"
            self.attempts[node_id] = int(event.payload.get("attempt", 0))

    @property
    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for state in self.status.values():
            tally[state] = tally.get(state, 0) + 1
        return tally

    def render(self) -> Table:
        table = Table(title="Omniord run", show_header=True, header_style="bold")
        table.add_column("node", style="cyan", no_wrap=True)
        table.add_column("status")
        for node_id, state in self.status.items():
            label, style = _STATUS_STYLE.get(state, (state, "white"))
            if state == "retrying" and node_id in self.attempts:
                label = f"retrying (#{self.attempts[node_id]})"
            table.add_row(node_id, f"[{style}]{label}[/{style}]")
        done_count = self.counts.get("completed", 0)
        table.caption = f"{done_count}/{self.total or len(self.status)} completed"
        return table
