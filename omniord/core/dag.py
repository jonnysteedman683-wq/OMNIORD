"""Task graph: typed nodes and a directed acyclic graph over them.

An ambiguous prompt is decomposed into a :class:`DAG` of :class:`TaskNode`
objects. The engine resolves execution order from the graph's dependencies via
topological sort and runs independent branches concurrently.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskNode(BaseModel):
    """A single unit of work in the task graph."""

    id: str
    description: str = ""
    dependencies: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class DAGError(Exception):
    """The graph is structurally invalid (e.g. a missing dependency)."""


class CycleError(DAGError):
    """The graph contains a cycle and has no valid execution order."""


class DAG:
    """A directed acyclic graph of :class:`TaskNode` objects, keyed by id."""

    def __init__(self, nodes: list[TaskNode] | None = None) -> None:
        self.nodes: dict[str, TaskNode] = {}
        for node in nodes or []:
            self.add(node)

    def add(self, node: TaskNode) -> TaskNode:
        if node.id in self.nodes:
            raise DAGError(f"duplicate node id: {node.id!r}")
        self.nodes[node.id] = node
        return node

    def add_node(
        self, id: str, description: str = "", dependencies: list[str] | None = None
    ) -> TaskNode:
        """Convenience constructor that builds and adds a node."""
        return self.add(
            TaskNode(id=id, description=description, dependencies=list(dependencies or []))
        )

    def dependents(self, node_id: str) -> list[str]:
        """Ids of nodes that depend directly on ``node_id``."""
        return [nid for nid, node in self.nodes.items() if node_id in node.dependencies]

    def validate(self) -> None:
        """Raise if any dependency is missing or the graph contains a cycle."""
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise DAGError(f"node {node.id!r} depends on unknown node {dep!r}")
                if dep == node.id:
                    raise CycleError(f"node {node.id!r} depends on itself")
        # A successful topological sort proves acyclicity.
        self.topological_order()

    def topological_order(self) -> list[str]:
        """Return a valid execution order (Kahn's algorithm).

        Raises :class:`CycleError` if no ordering exists.
        """
        indegree = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in indegree:
                    indegree[node.id] += 1
        # Sort the initial frontier for a deterministic order.
        ready = deque(sorted(nid for nid, d in indegree.items() if d == 0))
        order: list[str] = []
        while ready:
            nid = ready.popleft()
            order.append(nid)
            for dependent in sorted(self.dependents(nid)):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if len(order) != len(self.nodes):
            remaining = set(self.nodes) - set(order)
            raise CycleError(f"cycle detected among nodes: {sorted(remaining)}")
        return order

    def layers(self) -> list[list[str]]:
        """Group node ids into dependency layers.

        Every node in layer *n* depends only on nodes in layers < *n*, so a
        whole layer can be executed concurrently. Raises on a cycle.
        """
        indegree = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in indegree:
                    indegree[node.id] += 1
        frontier = sorted(nid for nid, d in indegree.items() if d == 0)
        seen = 0
        layers: list[list[str]] = []
        while frontier:
            layers.append(frontier)
            seen += len(frontier)
            nxt: list[str] = []
            for nid in frontier:
                for dependent in self.dependents(nid):
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        nxt.append(dependent)
            frontier = sorted(nxt)
        if seen != len(self.nodes):
            raise CycleError("cycle detected: not all nodes are reachable in layers")
        return layers
