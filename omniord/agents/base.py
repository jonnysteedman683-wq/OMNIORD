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
from collections.abc import Awaitable, Callable

from omniord.core.dag import TaskNode
from omniord.router.base import Message
from omniord.router.router import Router, TaskKind
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


class RouterAgent(BaseAgent):
    """An agent that executes a node by reasoning with the hybrid router.

    It builds a prompt from the node's description, prior dependency outputs, and
    any ``last_error`` left by the self-healing retry loop, then returns the
    model's text as the node output. Pure generation, so it declares no guarded
    action.
    """

    kind = "router"

    def __init__(
        self,
        router: Router,
        *,
        name: str | None = None,
        task: TaskKind = TaskKind.REASON,
        force: str | None = None,
    ):
        super().__init__(name)
        self.router = router
        self.task = task
        self.force = force

    async def run(self, node: TaskNode, context: dict) -> dict:
        parts = [f"Task: {node.description or node.id}"]
        deps = context.get("dependencies") or {}
        if deps:
            parts.append("Results from prior steps:")
            for dep_id, outputs in deps.items():
                parts.append(f"- {dep_id}: {outputs}")
        if context.get("last_error"):
            parts.append(
                f"Your previous attempt failed with: {context['last_error']}. "
                "Correct it this time."
            )
        result = await self.router.generate(
            [Message(role="user", content="\n".join(parts))],
            task=self.task,
            force=self.force,  # type: ignore[arg-type]
        )
        return {"text": result.text, "tier": result.tier}


def build_agent(kind: str, handler: Handler) -> BaseAgent:
    """Construct a typed agent by kind name, wrapping ``handler``."""
    by_kind: dict[str, type[_TypedAgent]] = {
        "coder": CoderAgent,
        "search": SearchAgent,
        "reviewer": ReviewerAgent,
        "sysadmin": SysAdminAgent,
    }
    return by_kind.get(kind, ReviewerAgent)(handler)
