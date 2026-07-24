"""Tests for the planner (prompt→DAG) and the router-backed agent."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from omniord.agents.base import RouterAgent
from omniord.config import OmniordSettings
from omniord.core.dag import TaskNode
from omniord.core.planner import Plan, Planner
from omniord.router.base import GenerationResult, LLMProvider, Message
from omniord.router.router import Router


class _ScriptedProvider(LLMProvider):
    """Local provider that returns a fixed text (for planner/agent tests)."""

    name = "scripted"
    tier = "local"

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    async def generate(self, messages: Sequence[Message], *, model: str | None = None) -> GenerationResult:
        self.prompts.append(messages[-1].content)
        return GenerationResult(text=self.text, provider=self.name, model=model or "m", tier=self.tier)

    async def stream(self, messages: Sequence[Message], *, model: str | None = None) -> AsyncIterator[str]:
        yield self.text

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def health_check(self) -> bool:
        return True


def _router(text: str) -> Router:
    return Router(OmniordSettings(_env_file=None), local=_ScriptedProvider(text), cloud=None)


async def test_planner_parses_json_into_dag() -> None:
    plan_json = """
    Here is your plan:
    [
      {"id": "gather", "description": "collect data", "dependencies": [], "agent": "search"},
      {"id": "write", "description": "draft report", "dependencies": ["gather"], "agent": "coder"}
    ]
    """
    plan = await Planner(_router(plan_json)).plan("write a report")
    assert isinstance(plan, Plan)
    assert [n.id for n in plan.nodes] == ["gather", "write"]
    assert plan.agent_kind("write") == "coder"
    dag = plan.to_dag()
    assert dag.topological_order() == ["gather", "write"]


async def test_planner_falls_back_to_single_node_on_bad_output() -> None:
    plan = await Planner(_router("sorry, no JSON here")).plan("do something")
    assert len(plan.nodes) == 1
    assert plan.nodes[0].description == "do something"


async def test_planner_normalizes_unknown_agent() -> None:
    plan = await Planner(_router('[{"id": "x", "agent": "wizard"}]')).plan("x")
    assert plan.nodes[0].agent == "reviewer"


async def test_router_agent_uses_context_and_returns_text() -> None:
    router = _router("the answer")
    agent = RouterAgent(router, name="n1")
    node = TaskNode(id="n1", description="answer the question")
    result = await agent.run(node, {"dependencies": {"dep": {"text": "prior"}}})
    assert result["text"] == "the answer"
    assert result["tier"] == "local"
    # The prompt should incorporate the dependency output.
    provider = router.local
    assert "prior" in provider.prompts[-1]  # type: ignore[attr-defined]
