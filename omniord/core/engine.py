"""Async execution engine for a task DAG.

The engine resolves dependencies from the graph and launches each node as soon
as all of its dependencies have completed — so independent branches run
concurrently. Each node receives its dependencies' outputs as context. Node
lifecycle transitions are published on an :class:`~omniord.core.events.EventBus`
for live progress rendering.

A node's work is supplied by a ``NodeExecutor`` coroutine::

    async def executor(node: TaskNode, context: dict[str, dict]) -> dict:
        ...  # context maps each dependency id -> that node's outputs
        return {"result": ...}  # becomes node.outputs
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from omniord.core.dag import DAG, NodeStatus, TaskNode
from omniord.core.events import EventBus, EventType

NodeExecutor = Callable[[TaskNode, dict[str, dict]], Awaitable[dict]]


class ExecutionEngine:
    """Runs a :class:`DAG` to completion, concurrently where possible."""

    def __init__(self, bus: EventBus | None = None, *, fail_fast: bool = False) -> None:
        self.bus = bus or EventBus()
        self.fail_fast = fail_fast

    async def run(self, dag: DAG, executor: NodeExecutor) -> dict[str, TaskNode]:
        """Execute every node in ``dag`` and return the nodes with final state.

        A node whose dependency failed (or was skipped) is marked
        ``SKIPPED``; independent branches keep running. With ``fail_fast`` the
        engine cancels in-flight nodes and skips the rest on the first failure.
        """
        dag.validate()
        await self.bus.publish(EventType.DAG_STARTED, total=len(dag.nodes))

        completed: set[str] = set()
        blocked: set[str] = set()  # failed or skipped — dependents can't run
        started: set[str] = set()
        running: dict[asyncio.Task, str] = {}

        async def schedule() -> None:
            """Launch every newly-ready node and skip every newly-blocked one.

            Skips cascade, so repeat until a full pass changes nothing.
            """
            while True:
                changed = False
                for nid, node in dag.nodes.items():
                    if nid in started:
                        continue
                    deps = node.dependencies
                    if any(dep in blocked for dep in deps):
                        node.status = NodeStatus.SKIPPED
                        started.add(nid)
                        blocked.add(nid)
                        await self.bus.publish(EventType.NODE_SKIPPED, id=nid)
                        changed = True
                    elif all(dep in completed for dep in deps):
                        node.status = NodeStatus.RUNNING
                        started.add(nid)
                        await self.bus.publish(EventType.NODE_RUNNING, id=nid)
                        context = {dep: dag.nodes[dep].outputs for dep in deps}
                        task = asyncio.create_task(executor(node, context))
                        running[task] = nid
                        changed = True
                if not changed:
                    return

        await schedule()
        while running:
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                nid = running.pop(task)
                node = dag.nodes[nid]
                try:
                    outputs = task.result()
                    node.outputs = outputs or {}
                    node.status = NodeStatus.COMPLETED
                    completed.add(nid)
                    await self.bus.publish(EventType.NODE_COMPLETED, id=nid)
                except Exception as exc:
                    node.status = NodeStatus.FAILED
                    node.error = str(exc)
                    blocked.add(nid)
                    await self.bus.publish(EventType.NODE_FAILED, id=nid, error=str(exc))
                    if self.fail_fast:
                        await self._abort(dag, running, started, blocked)
                        await self.bus.publish(
                            EventType.DAG_COMPLETED, completed=len(completed), aborted=True
                        )
                        return dag.nodes
            await schedule()

        await self.bus.publish(
            EventType.DAG_COMPLETED, completed=len(completed), aborted=False
        )
        return dag.nodes

    async def _abort(
        self,
        dag: DAG,
        running: dict[asyncio.Task, str],
        started: set[str],
        blocked: set[str],
    ) -> None:
        """Cancel in-flight tasks and mark everything unfinished as skipped."""
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        running.clear()
        for nid, node in dag.nodes.items():
            if node.status in (NodeStatus.PENDING, NodeStatus.RUNNING):
                node.status = NodeStatus.SKIPPED
                started.add(nid)
                blocked.add(nid)
