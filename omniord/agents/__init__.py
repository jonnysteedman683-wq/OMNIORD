"""Ephemeral agent swarm: task-focused workers that run DAG nodes concurrently."""

from __future__ import annotations

from omniord.agents.base import (
    BaseAgent,
    CoderAgent,
    FunctionAgent,
    ReviewerAgent,
    RouterAgent,
    SearchAgent,
    SysAdminAgent,
    build_agent,
)
from omniord.agents.swarm import Swarm, SwarmError

__all__ = [
    "BaseAgent",
    "CoderAgent",
    "FunctionAgent",
    "ReviewerAgent",
    "RouterAgent",
    "SearchAgent",
    "Swarm",
    "SwarmError",
    "SysAdminAgent",
    "build_agent",
]
