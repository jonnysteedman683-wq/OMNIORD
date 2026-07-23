"""Working memory: a thread-safe in-memory scratchpad for a run.

Agents in the swarm share one :class:`WorkingMemory` so a node's results are
visible to later nodes and to sibling workers. It holds task state for the
duration of a DAG execution; durable, cross-session memory lives in the
persistent store (Phase 6).
"""

from __future__ import annotations

import threading
from typing import Any


class WorkingMemory:
    """A lock-guarded key/value store shared across concurrent workers."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def update(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(values)

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the current contents."""
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
