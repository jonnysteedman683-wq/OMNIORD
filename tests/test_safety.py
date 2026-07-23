"""Phase 5 tests: safety guardrails, working memory, and the agent swarm."""

from __future__ import annotations

import pytest

from omniord.agents.base import (
    CoderAgent,
    FunctionAgent,
    ReviewerAgent,
    SysAdminAgent,
)
from omniord.agents.swarm import Swarm, SwarmError
from omniord.core.dag import DAG, NodeStatus, TaskNode
from omniord.memory.working import WorkingMemory
from omniord.safety.guard import (
    Action,
    ActionDenied,
    RiskAssessor,
    RiskLevel,
    SafetyGuard,
    build_diff,
)


# --------------------------------------------------------------------------- #
# Risk assessment
# --------------------------------------------------------------------------- #


def test_assessor_classifies_by_kind() -> None:
    assessor = RiskAssessor()
    assert assessor.assess(Action(kind="read")).level is RiskLevel.SAFE
    assert assessor.assess(Action(kind="write_file")).level is RiskLevel.MODERATE
    assert assessor.assess(Action(kind="delete")).level is RiskLevel.CRITICAL


def test_assessor_flags_destructive_commands() -> None:
    assessor = RiskAssessor()
    action = Action(kind="task", command="rm -rf /important")
    assert assessor.assess(action).level is RiskLevel.CRITICAL


def test_unclassified_defaults_to_moderate() -> None:
    assert RiskAssessor().assess(Action(kind="mystery")).level is RiskLevel.MODERATE


def test_build_diff() -> None:
    diff = build_diff(Action(kind="edit_file", target="f.txt", old="a\nb\n", new="a\nc\n"))
    assert diff is not None
    assert "-b" in diff and "+c" in diff
    assert build_diff(Action(kind="read")) is None


# --------------------------------------------------------------------------- #
# Guard decisions
# --------------------------------------------------------------------------- #


async def test_safe_action_auto_executes() -> None:
    decision = await SafetyGuard().review(Action(kind="read"))
    assert decision.allowed is True
    assert decision.auto is True
    assert decision.level is RiskLevel.SAFE


async def test_moderate_action_logs_and_allows() -> None:
    notices: list[str] = []
    guard = SafetyGuard(on_notice=notices.append)
    decision = await guard.review(Action(kind="write_file", target="out.txt"))
    assert decision.allowed is True
    assert any("moderate" in n for n in notices)


async def test_critical_denied_without_handler() -> None:
    decision = await SafetyGuard().review(Action(kind="delete", target="db"))
    assert decision.allowed is False
    assert "no confirmation handler" in decision.reason


async def test_critical_requires_confirmation() -> None:
    approving = SafetyGuard(confirm=lambda a, r: True)
    denying = SafetyGuard(confirm=lambda a, r: False)
    assert (await approving.review(Action(kind="shell", command="ls"))).allowed is True
    assert (await denying.review(Action(kind="shell", command="ls"))).allowed is False


async def test_critical_accepts_async_confirmation() -> None:
    async def confirm(action: Action, assessment) -> bool:
        return True

    decision = await SafetyGuard(confirm=confirm).review(Action(kind="delete"))
    assert decision.allowed is True


async def test_enforce_raises_on_denial() -> None:
    with pytest.raises(ActionDenied):
        await SafetyGuard().enforce(Action(kind="delete"))


# --------------------------------------------------------------------------- #
# Working memory
# --------------------------------------------------------------------------- #


def test_working_memory_basics() -> None:
    memory = WorkingMemory()
    memory.set("a", 1)
    memory.update({"b": 2})
    assert memory.get("a") == 1
    assert "b" in memory
    assert memory.snapshot() == {"a": 1, "b": 2}
    assert len(memory) == 2
    memory.clear()
    assert len(memory) == 0


# --------------------------------------------------------------------------- #
# Swarm
# --------------------------------------------------------------------------- #


async def _echo(node: TaskNode, context: dict) -> dict:
    return {"handled_by": node.id}


async def test_swarm_runs_nodes_and_shares_memory() -> None:
    dag = DAG()
    dag.add_node("n1")
    dag.add_node("n2", dependencies=["n1"])
    swarm = Swarm()
    swarm.assign("n1", ReviewerAgent(_echo))  # safe
    swarm.assign("n2", ReviewerAgent(_echo))
    nodes = await swarm.run(dag)
    assert all(n.status is NodeStatus.COMPLETED for n in nodes.values())
    assert swarm.memory.get("n1") == {"handled_by": "n1"}
    assert swarm.memory.get("n2") == {"handled_by": "n2"}


async def test_swarm_requires_agent_for_every_node() -> None:
    dag = DAG([TaskNode(id="orphan")])
    with pytest.raises(SwarmError):
        await Swarm().run(dag)


async def test_swarm_blocks_critical_action_without_confirmation() -> None:
    dag = DAG([TaskNode(id="danger")])
    swarm = Swarm()  # default guard has no confirm handler → critical denied
    swarm.assign("danger", SysAdminAgent(_echo))  # declares a shell (critical) action
    nodes = await swarm.run(dag)
    assert nodes["danger"].status is NodeStatus.FAILED
    assert "denied" in (nodes["danger"].error or "")


async def test_swarm_allows_critical_action_with_confirmation() -> None:
    dag = DAG([TaskNode(id="danger")])
    swarm = Swarm(guard=SafetyGuard(confirm=lambda a, r: True))
    swarm.assign("danger", SysAdminAgent(_echo))
    nodes = await swarm.run(dag)
    assert nodes["danger"].status is NodeStatus.COMPLETED


async def test_swarm_denied_action_skips_dependents() -> None:
    dag = DAG()
    dag.add_node("danger")
    dag.add_node("downstream", dependencies=["danger"])
    swarm = Swarm()
    swarm.assign("danger", SysAdminAgent(_echo))    # critical, denied
    swarm.assign("downstream", ReviewerAgent(_echo))
    nodes = await swarm.run(dag)
    assert nodes["danger"].status is NodeStatus.FAILED
    assert nodes["downstream"].status is NodeStatus.SKIPPED


async def test_swarm_tears_down_agents() -> None:
    torn: list[str] = []

    class TrackingAgent(FunctionAgent):
        async def teardown(self) -> None:
            torn.append(self.name)

    dag = DAG([TaskNode(id="n1")])
    swarm = Swarm()
    swarm.assign("n1", TrackingAgent(_echo, name="worker-1"))
    await swarm.run(dag)
    assert torn == ["worker-1"]


async def test_swarm_selector_assigns_by_node() -> None:
    dag = DAG([TaskNode(id="a"), TaskNode(id="b")])
    swarm = Swarm(selector=lambda node: ReviewerAgent(_echo, name=f"agent-{node.id}"))
    nodes = await swarm.run(dag)
    assert all(n.status is NodeStatus.COMPLETED for n in nodes.values())


async def test_coder_agent_is_moderate_and_runs() -> None:
    # A moderate action executes without a confirmation handler.
    dag = DAG([TaskNode(id="code")])
    swarm = Swarm()
    swarm.assign("code", CoderAgent(_echo))
    nodes = await swarm.run(dag)
    assert nodes["code"].status is NodeStatus.COMPLETED
