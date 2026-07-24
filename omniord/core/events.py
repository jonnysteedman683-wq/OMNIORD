"""A small async pub/sub event bus for real-time execution state.

The execution engine publishes typed events (node started, completed, failed,
…) that a terminal UI — or any other subscriber — can render live. Handlers may
be sync or async; a handler that raises is isolated so it cannot take down the
publisher or the other subscribers.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

Handler = Callable[["Event"], Any] | Callable[["Event"], Awaitable[Any]]


class EventType(str, Enum):
    """Well-known event types emitted by the execution engine."""

    DAG_STARTED = "dag_started"
    DAG_COMPLETED = "dag_completed"
    NODE_RUNNING = "node_running"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    NODE_RETRY = "node_retry"


class Event(BaseModel):
    """A single event with an arbitrary payload."""

    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)


class EventBus:
    """Dispatches events to subscribed handlers."""

    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def subscribe(self, handler: Handler) -> Callable[[], None]:
        """Register ``handler``; returns a callable that unsubscribes it."""
        self._handlers.append(handler)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._handlers.remove(handler)

        return unsubscribe

    async def publish(self, type: str | EventType, **payload: Any) -> Event:
        """Build and dispatch an event to every subscriber."""
        type_value = type.value if isinstance(type, EventType) else type
        event = Event(type=str(type_value), payload=payload)
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # A misbehaving subscriber must not break publishing.
                continue
        return event

    @contextlib.asynccontextmanager
    async def stream(self) -> AsyncIterator[asyncio.Queue[Event]]:
        """Yield a queue that receives every event for the context's duration.

        Useful for driving a live progress display::

            async with bus.stream() as queue:
                event = await queue.get()
        """
        queue: asyncio.Queue[Event] = asyncio.Queue()
        unsubscribe = self.subscribe(queue.put_nowait)
        try:
            yield queue
        finally:
            unsubscribe()
