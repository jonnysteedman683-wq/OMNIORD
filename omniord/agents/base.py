"""Base agent interface and a set of task-focused worker agents.

An agent does the work of a single DAG node. Before running, it may declare the
side-effecting :class:`~omniord.safety.guard.Action` it intends to perform, which
the swarm routes through the :class:`~omniord.safety.guard.SafetyGuard`. Agents
are ephemeral: the swarm tears each one down after its node completes.

Concrete agents wrap an injected async handler so behavior stays testable and
free of hardcoded I/O; they differ by their ``kind`` and the risk class of the
action they declare (a ``SysAdminAgent`` runs shell → critical; a ``SearchAgent``
fetches → moderate; a ``ReviewerAgent`` reads → safe).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from omniord.core.dag import TaskNode
from omniord.safety.guard import Action

Handler = Callable[[TaskNode, dict], Awaitable[dict]]


class BaseAgent(ABC):
    """Abstract worker. Subclasses implement :meth:`run`."""

    #: Category of work this agent performs (used for logging/selection).
    kind: str = "agent"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def action_for(self, node: TaskNode, context: dict) -> Action | None:
        """Declare the guarded action this node will perform, or None if the
        work has no side effects worth reviewing."""
        return None

    @abstractmethod
    async def run(self, node: TaskNode, context: dict) -> dict:
        """Perform the node's work and return its outputs."""

    async def teardown(self) -> None:
        """Release resources. Called by the swarm after the node completes."""
        return None


class FunctionAgent(BaseAgent):
    """A generic agent that delegates to an injected async handler.

    ``action_kind`` (with optional ``action_factory``) controls how the swarm's
    guard classifies the work.
    """

    kind = "function"

    def __init__(
        self,
        handler: Handler,
        *,
        name: str | None = None,
        action_kind: str | None = None,
        action_factory: Callable[[TaskNode, dict], Action | None] | None = None,
    ) -> None:
        super().__init__(name)
        self._handler = handler
        self._action_kind = action_kind
        self._action_factory = action_factory

    def action_for(self, node: TaskNode, context: dict) -> Action | None:
        if self._action_factory is not None:
            return self._action_factory(node, context)
        if self._action_kind is not None:
            return Action(kind=self._action_kind, description=node.description, target=node.id)
        return None

    async def run(self, node: TaskNode, context: dict) -> dict:
        return await self._handler(node, context)


class _TypedAgent(FunctionAgent):
    """A FunctionAgent with a fixed kind and default declared action kind."""

    kind = "typed"
    default_action_kind: str | None = None

    def __init__(self, handler: Handler, *, name: str | None = None) -> None:
        super().__init__(handler, name=name, action_kind=self.default_action_kind)


class CoderAgent(_TypedAgent):
    kind = "coder"
    default_action_kind = "write_file"  # moderate


class SearchAgent(_TypedAgent):
    kind = "search"
    default_action_kind = "api_fetch"  # moderate


class ReviewerAgent(_TypedAgent):
    kind = "reviewer"
    default_action_kind = "read"  # safe


class SysAdminAgent(_TypedAgent):
    kind = "sysadmin"
    default_action_kind = "shell"  # critical
