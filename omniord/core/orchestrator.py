"""The core orchestrator: memory recall → guarded swarm execution → persistence.

This ties the subsystems together. For a task it (1) recalls the most relevant
prior episodes from the persistent :class:`~omniord.memory.store.MemoryStore` and
seeds them into the swarm's shared working memory, (2) runs the task DAG through
the :class:`~omniord.agents.swarm.Swarm` (concurrent, guarded execution), and
(3) persists the run's outcome back to the store for future recall.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from omniord.agents.swarm import Swarm
from omniord.core.dag import DAG, NodeStatus, TaskNode
from omniord.core.events import EventBus
from omniord.memory.store import MemoryRecord, MemoryStore


class OrchestrationResult(BaseModel):
    nodes: dict[str, TaskNode]
    recalled: list[MemoryRecord] = Field(default_factory=list)
    persisted_id: str | None = None

    @property
    def succeeded(self) -> bool:
        return all(n.status is NodeStatus.COMPLETED for n in self.nodes.values())


class Orchestrator:
    """Coordinates memory, the agent swarm, and outcome persistence."""

    def __init__(
        self,
        memory: MemoryStore,
        *,
        swarm: Swarm | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.memory = memory
        self.bus = bus or EventBus()
        self.swarm = swarm or Swarm(bus=self.bus)

    async def recall(self, query: str, k: int = 5) -> list[tuple[MemoryRecord, float]]:
        """Return prior episodes most relevant to ``query``."""
        if not query:
            return []
        return await self.memory.search(query, k)

    async def run(
        self, dag: DAG, *, task: str = "", recall_k: int = 5, persist: bool = True
    ) -> OrchestrationResult:
        """Recall context, execute the DAG via the swarm, and persist the outcome."""
        recalled = await self.recall(task, recall_k) if task else []
        self.swarm.memory.set("_recall", [record.text for record, _ in recalled])

        nodes = await self.swarm.run(dag)

        persisted_id: str | None = None
        if persist and task:
            outcome = {
                nid: node.outputs
                for nid, node in nodes.items()
                if node.status is NodeStatus.COMPLETED
            }
            persisted_id = await self.memory.add(
                task, metadata={"outcome": outcome, "succeeded": self._succeeded(nodes)}
            )

        return OrchestrationResult(
            nodes=nodes,
            recalled=[record for record, _ in recalled],
            persisted_id=persisted_id,
        )

    @staticmethod
    def _succeeded(nodes: dict[str, TaskNode]) -> bool:
        return all(n.status is NodeStatus.COMPLETED for n in nodes.values())
