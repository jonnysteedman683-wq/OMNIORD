"""Intent parser / task decomposition: turn a prompt into a task DAG.

The planner asks the router's fast local tier (Tier 0) to decompose an ambiguous
prompt into a list of typed steps with dependencies, then builds and validates a
:class:`~omniord.core.dag.DAG` from them. If the model's output can't be parsed,
it falls back to a single-node plan so execution can still proceed.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from omniord.core.dag import DAG
from omniord.router.base import Message
from omniord.router.router import Router, TaskKind

# Agent kinds the planner may assign to a step (matches agents.base).
KNOWN_AGENTS = {"reviewer", "coder", "search", "sysadmin"}


class PlanNode(BaseModel):
    id: str
    description: str = ""
    dependencies: list[str] = Field(default_factory=list)
    agent: str = "reviewer"


class Plan(BaseModel):
    prompt: str
    nodes: list[PlanNode]

    def to_dag(self) -> DAG:
        dag = DAG()
        for node in self.nodes:
            dag.add_node(node.id, node.description, node.dependencies)
        dag.validate()
        return dag

    def agent_kind(self, node_id: str) -> str:
        for node in self.nodes:
            if node.id == node_id:
                return node.agent
        return "reviewer"


_SYSTEM = (
    "You are a task-planning assistant. Decompose the user's request into a "
    "minimal directed acyclic graph of steps. Respond with ONLY a JSON array; "
    "each element is an object with keys: id (short slug), description, "
    "dependencies (array of ids), and agent (one of: reviewer, coder, search, "
    "sysadmin). Dependencies must reference earlier ids and must not form a cycle."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class Planner:
    """Decomposes a prompt into a :class:`Plan` using the router's Tier-0 model."""

    def __init__(self, router: Router) -> None:
        self.router = router

    async def plan(self, prompt: str, *, force: str | None = None) -> Plan:
        result = await self.router.generate(
            [Message(role="system", content=_SYSTEM), Message(role="user", content=prompt)],
            task=TaskKind.DAG,
            force=force,  # type: ignore[arg-type]
        )
        nodes = self._parse(result.text)
        if not nodes:
            nodes = [PlanNode(id="task", description=prompt, agent="reviewer")]
        return Plan(prompt=prompt, nodes=nodes)

    @staticmethod
    def _parse(text: str) -> list[PlanNode]:
        match = _JSON_ARRAY_RE.search(text)
        if not match:
            return []
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        nodes: list[PlanNode] = []
        for item in raw:
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                node = PlanNode.model_validate(item)
            except ValidationError:
                continue
            if node.agent not in KNOWN_AGENTS:
                node.agent = "reviewer"
            nodes.append(node)
        return nodes
