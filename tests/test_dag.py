"""Phase 3 tests: the DAG, the async execution engine, and the event bus."""

from __future__ import annotations

import asyncio

import pytest

from omniord.core.dag import DAG, CycleError, DAGError, NodeStatus, TaskNode
from omniord.core.engine import ExecutionEngine
from omniord.core.events import Event, EventBus, EventType


# --------------------------------------------------------------------------- #
# DAG structure
# --------------------------------------------------------------------------- #


def _diamond() -> DAG:
    dag = DAG()
    dag.add_node("root")
    dag.add_node("a", dependencies=["root"])
    dag.add_node("b", dependencies=["root"])
    dag.add_node("sink", dependencies=["a", "b"])
    return dag


def test_node_defaults() -> None:
    node = TaskNode(id="n1", description="do a thing")
    assert node.status is NodeStatus.PENDING
    assert node.dependencies == []
    assert node.outputs == {}
    assert node.error is None


def test_duplicate_id_rejected() -> None:
    dag = DAG()
    dag.add_node("n1")
    with pytest.raises(DAGError):
        dag.add_node("n1")


def test_missing_dependency_rejected() -> None:
    dag = DAG([TaskNode(id="n1", dependencies=["ghost"])])
    with pytest.raises(DAGError):
        dag.validate()


def test_self_dependency_is_a_cycle() -> None:
    dag = DAG([TaskNode(id="n1", dependencies=["n1"])])
    with pytest.raises(CycleError):
        dag.validate()


def test_cycle_detected() -> None:
    dag = DAG(
        [
            TaskNode(id="a", dependencies=["b"]),
            TaskNode(id="b", dependencies=["a"]),
        ]
    )
    with pytest.raises(CycleError):
        dag.topological_order()


def test_topological_order_respects_dependencies() -> None:
    order = _diamond().topological_order()
    assert order.index("root") < order.index("a")
    assert order.index("root") < order.index("b")
    assert order.index("a") < order.index("sink")
    assert order.index("b") < order.index("sink")


def test_layers_group_independent_nodes() -> None:
    assert _diamond().layers() == [["root"], ["a", "b"], ["sink"]]


def test_dependents() -> None:
    assert sorted(_diamond().dependents("root")) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Execution engine
# --------------------------------------------------------------------------- #


async def test_linear_execution_order() -> None:
    dag = DAG()
    dag.add_node("n1")
    dag.add_node("n2", dependencies=["n1"])
    dag.add_node("n3", dependencies=["n2"])
    order: list[str] = []

    async def run_node(node: TaskNode, ctx: dict[str, dict]) -> dict:
        order.append(node.id)
        return {"ok": True}

    nodes = await ExecutionEngine().run(dag, run_node)
    assert order == ["n1", "n2", "n3"]
    assert all(n.status is NodeStatus.COMPLETED for n in nodes.values())


async def test_outputs_propagate_through_context() -> None:
    dag = _diamond()

    async def run_node(node: TaskNode, ctx: dict[str, dict]) -> dict:
        incoming = sum(out.get("val", 0) for out in ctx.values())
        return {"val": incoming + 1}

    nodes = await ExecutionEngine().run(dag, run_node)
    # root=1, a=2, b=2, sink=(2+2)+1=5
    assert nodes["root"].outputs["val"] == 1
    assert nodes["a"].outputs["val"] == 2
    assert nodes["sink"].outputs["val"] == 5


async def test_independent_nodes_run_concurrently() -> None:
    dag = DAG([TaskNode(id=f"n{i}") for i in range(3)])
    state = {"current": 0, "peak": 0}

    async def run_node(node: TaskNode, ctx: dict[str, dict]) -> dict:
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await asyncio.sleep(0.05)
        state["current"] -= 1
        return {}

    await ExecutionEngine().run(dag, run_node)
    assert state["peak"] == 3


async def test_failure_skips_dependents_but_not_independents() -> None:
    dag = DAG()
    dag.add_node("a")
    dag.add_node("b", dependencies=["a"])  # depends on failing node
    dag.add_node("c", dependencies=["b"])  # transitively blocked
    dag.add_node("d")  # independent, should still run

    async def run_node(node: TaskNode, ctx: dict[str, dict]) -> dict:
        if node.id == "a":
            raise RuntimeError("a failed")
        return {}

    nodes = await ExecutionEngine().run(dag, run_node)
    assert nodes["a"].status is NodeStatus.FAILED
    assert nodes["a"].error == "a failed"
    assert nodes["b"].status is NodeStatus.SKIPPED
    assert nodes["c"].status is NodeStatus.SKIPPED
    assert nodes["d"].status is NodeStatus.COMPLETED


async def test_fail_fast_aborts_inflight_work() -> None:
    dag = DAG([TaskNode(id="fast"), TaskNode(id="slow")])

    async def run_node(node: TaskNode, ctx: dict[str, dict]) -> dict:
        if node.id == "fast":
            raise RuntimeError("boom")
        await asyncio.sleep(1.0)  # would outlast the test if not cancelled
        return {}

    nodes = await ExecutionEngine(fail_fast=True).run(dag, run_node)
    assert nodes["fast"].status is NodeStatus.FAILED
    assert nodes["slow"].status is NodeStatus.SKIPPED


# --------------------------------------------------------------------------- #
# Event bus
# --------------------------------------------------------------------------- #


async def test_engine_publishes_lifecycle_events() -> None:
    dag = DAG([TaskNode(id="only")])
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda e: seen.append(e.type))

    async def run_node(node: TaskNode, ctx: dict[str, dict]) -> dict:
        return {}

    await ExecutionEngine(bus).run(dag, run_node)
    assert seen == [
        EventType.DAG_STARTED.value,
        EventType.NODE_RUNNING.value,
        EventType.NODE_COMPLETED.value,
        EventType.DAG_COMPLETED.value,
    ]


async def test_bus_supports_sync_and_async_handlers_and_isolates_errors() -> None:
    bus = EventBus()
    received: list[Event] = []

    def bad(_: Event) -> None:
        raise ValueError("subscriber blew up")

    async def good(event: Event) -> None:
        received.append(event)

    bus.subscribe(bad)
    bus.subscribe(good)
    await bus.publish("custom", value=42)
    assert len(received) == 1
    assert received[0].type == "custom"
    assert received[0].payload == {"value": 42}


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    hits: list[Event] = []
    unsubscribe = bus.subscribe(hits.append)
    await bus.publish("one")
    unsubscribe()
    await bus.publish("two")
    assert [e.type for e in hits] == ["one"]


async def test_stream_context_receives_events() -> None:
    bus = EventBus()
    async with bus.stream() as queue:
        await bus.publish(EventType.NODE_RUNNING, id="x")
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event.type == EventType.NODE_RUNNING.value
    assert event.payload == {"id": "x"}
