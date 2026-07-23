"""Swarm orchestrator: run DAG nodes with worker agents, guarded and concurrent.

The swarm assigns an agent to each node and drives the graph through the Phase-3
:class:`~omniord.core.engine.ExecutionEngine`, so independent nodes run
concurrently. Around every worker step it:

1. builds the node's context (dependency outputs + shared working-memory snapshot),
2. asks the agent to declare its side-effecting action and routes it through the
   :class:`~omniord.safety.guard.SafetyGuard` — a denied action fails the node
   (its dependents are then skipped by the engine),
3. runs the agent, records its outputs in the shared
   :class:`~omniord.memory.working.WorkingMemory`, and
4. tears the agent down (agents are ephemeral).
"""

from __future__ import annotations

from collections.abc import Callable

from omniord.agents.base import BaseAgent
from omniord.core.dag import DAG, TaskNode
from omniord.core.engine import ExecutionEngine
from omniord.core.events import EventBus, EventType
from omniord.memory.working import WorkingMemory
from omniord.safety.guard import ActionDenied, SafetyGuard

AgentSelector = Callable[[TaskNode], BaseAgent]


class SwarmError(Exception):
    """The swarm was misconfigured (e.g. a node has no assigned agent)."""


class Swarm:
    """Coordinates worker agents over a task DAG."""

    def __init__(
        self,
        *,
        engine: ExecutionEngine | None = None,
        guard: SafetyGuard | None = None,
        memory: WorkingMemory | None = None,
        bus: EventBus | None = None,
        selector: AgentSelector | None = None,
        max_retries: int = 0,
    ) -> None:
        self.bus = bus or EventBus()
        self.engine = engine or ExecutionEngine(self.bus)
        self.guard = guard or SafetyGuard()
        self.memory = memory or WorkingMemory()
        self.max_retries = max_retries
        self._assignments: dict[str, BaseAgent] = {}
        self._selector = selector

    def assign(self, node_id: str, agent: BaseAgent) -> None:
        """Bind ``agent`` to the node with id ``node_id``."""
        self._assignments[node_id] = agent

    def _agent_for(self, node: TaskNode) -> BaseAgent:
        agent = self._assignments.get(node.id)
        if agent is not None:
            return agent
        if self._selector is not None:
            return self._selector(node)
        raise SwarmError(f"no agent assigned for node {node.id!r}")

    async def run(self, dag: DAG) -> dict[str, TaskNode]:
        # Fail early on misconfiguration rather than mid-execution.
        for node in dag.nodes.values():
            self._agent_for(node)

        async def execute(node: TaskNode, dep_outputs: dict[str, dict]) -> dict:
            agent = self._agent_for(node)
            context: dict = {"dependencies": dep_outputs, "memory": self.memory.snapshot()}
            attempt = 0
            try:
                # Self-healing loop: an action denial is terminal, but any other
                # failure is retried (with the error fed back into context) up to
                # max_retries times before it propagates and fails the node.
                while True:
                    attempt += 1
                    try:
                        action = agent.action_for(node, context)
                        if action is not None:
                            await self.guard.enforce(action)
                        outputs = await agent.run(node, context)
                        break
                    except ActionDenied:
                        raise
                    except Exception as exc:
                        if attempt > self.max_retries:
                            raise
                        context = {**context, "last_error": str(exc), "attempt": attempt}
                        await self.bus.publish(
                            EventType.NODE_RETRY, id=node.id, attempt=attempt, error=str(exc)
                        )
            finally:
                await agent.teardown()
            self.memory.set(node.id, outputs)
            return outputs

        return await self.engine.run(dag, execute)
