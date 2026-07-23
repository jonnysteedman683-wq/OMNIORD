"""Phase 6 tests: persistent memory store, orchestrator, and end-to-end pipeline.

The end-to-end test exercises the whole stack: the tool factory synthesizes a
tool, the swarm runs it in the sandbox as a DAG, the result is saved to the
persistent memory store, and it is then retrieved by semantic search.
"""

from __future__ import annotations

from pathlib import Path

from omniord.agents.base import CoderAgent, FunctionAgent, ReviewerAgent
from omniord.agents.swarm import Swarm
from omniord.core.dag import DAG, NodeStatus, TaskNode
from omniord.core.orchestrator import Orchestrator
from omniord.memory.store import HashingEmbedder, MemoryStore
from omniord.safety.guard import SafetyGuard
from omniord.tools.factory import ToolFactory, ToolSpec
from omniord.tools.registry import ToolRegistry
from omniord.tools.sandbox import Sandbox


# --------------------------------------------------------------------------- #
# HashingEmbedder + MemoryStore
# --------------------------------------------------------------------------- #


async def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    a = (await embedder.embed(["hello world"]))[0]
    b = (await embedder.embed(["hello world"]))[0]
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9  # unit length


async def test_memory_add_get_and_count(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.db")
    rec_id = await store.add("remember the milk", metadata={"tag": "todo"})
    assert store.count() == 1
    record = store.get(rec_id)
    assert record is not None
    assert record.text == "remember the milk"
    assert record.metadata == {"tag": "todo"}
    store.close()


async def test_memory_semantic_search(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.db")
    await store.add("the cat sat on the mat")
    await store.add("python async programming with asyncio")
    results = await store.search("cat", k=3)
    assert results
    assert "cat" in results[0][0].text
    store.close()


async def test_memory_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "mem.db"
    store = MemoryStore(path)
    await store.add("durable knowledge about widgets")
    store.close()

    reopened = MemoryStore(path)
    assert reopened.count() == 1
    results = await reopened.search("widgets", k=1)
    assert results and "widgets" in results[0][0].text
    reopened.close()


async def test_memory_delete_and_where_filter(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.db")
    id_a = await store.add("alpha note", metadata={"kind": "a"})
    await store.add("alpha note", metadata={"kind": "b"})
    filtered = await store.search("alpha", k=5, where={"kind": "a"})
    assert len(filtered) == 1
    assert filtered[0][0].metadata["kind"] == "a"

    assert store.delete(id_a) is True
    assert store.count() == 1
    store.close()


# --------------------------------------------------------------------------- #
# Orchestrator: recall + persist
# --------------------------------------------------------------------------- #


async def _noop(node: TaskNode, context: dict) -> dict:
    return {"seen_recall": context["memory"].get("_recall")}


async def test_orchestrator_persists_and_recalls(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.db")
    orch = Orchestrator(store)

    dag = DAG([TaskNode(id="step")])
    orch.swarm.assign("step", ReviewerAgent(_noop))
    result = await orch.run(dag, task="summarize the quarterly report")
    assert result.succeeded
    assert result.persisted_id is not None
    assert store.count() == 1

    # A later, similar task recalls the earlier one.
    dag2 = DAG([TaskNode(id="step")])
    orch.swarm.assign("step", ReviewerAgent(_noop))
    result2 = await orch.run(dag2, task="summarize the quarterly report again")
    assert any("quarterly report" in r.text for r in result2.recalled)
    store.close()


# --------------------------------------------------------------------------- #
# End-to-end: build a tool, run it, save the result, retrieve it
# --------------------------------------------------------------------------- #


class _FixedGenerator:
    def __init__(self, code: str) -> None:
        self.code = code

    async def generate(self, spec: ToolSpec) -> str:
        return self.code

    async def repair(self, spec: ToolSpec, code: str, error: str) -> str:
        return self.code


async def test_end_to_end_tool_build_run_remember_recall(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "tools")
    factory = ToolFactory(
        _FixedGenerator("def multiply(a, b):\n    return a * b\n"),
        registry,
    )
    store = MemoryStore(tmp_path / "mem.db")
    sandbox = Sandbox()

    async def build_tool(node: TaskNode, context: dict) -> dict:
        spec = ToolSpec(name="multiply", validation_code="assert multiply(6, 7) == 42\n")
        result = await factory.build(spec)
        assert result.success and result.registered
        return {"tool": "multiply"}

    async def run_tool(node: TaskNode, context: dict) -> dict:
        tool = context["dependencies"]["build_tool"]["tool"]
        source = registry.load_source(tool) + "\nprint(multiply(6, 7))\n"
        run = await sandbox.run(source)
        assert run.ok, run.stderr
        return {"result": int(run.stdout.strip())}

    async def remember(node: TaskNode, context: dict) -> dict:
        value = context["dependencies"]["run_tool"]["result"]
        rec_id = await store.add(f"multiply(6, 7) computed to {value}", metadata={"value": value})
        return {"stored": rec_id}

    dag = DAG()
    dag.add_node("build_tool")
    dag.add_node("run_tool", dependencies=["build_tool"])
    dag.add_node("remember", dependencies=["run_tool"])

    swarm = Swarm(guard=SafetyGuard(confirm=lambda a, r: True))
    swarm.assign("build_tool", CoderAgent(build_tool))
    swarm.assign("run_tool", FunctionAgent(run_tool, action_kind="read"))
    swarm.assign("remember", FunctionAgent(remember, action_kind="write"))

    nodes = await swarm.run(dag)
    assert all(n.status is NodeStatus.COMPLETED for n in nodes.values()), {
        nid: (n.status, n.error) for nid, n in nodes.items()
    }
    assert nodes["run_tool"].outputs["result"] == 42

    # The tool persisted to the registry, and the result is recallable.
    assert registry.get("multiply") is not None
    hits = await store.search("what did multiply compute", k=1)
    assert hits and hits[0][0].metadata["value"] == 42
    store.close()
