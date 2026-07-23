"""Action risk assessment and the interactive safety guard.

Every side-effecting action is classified before it runs:

* ``SAFE`` (read-only / math / in-memory) → auto-execute.
* ``MODERATE`` (file creation / API fetch) → log to the console and execute.
* ``CRITICAL`` (system mutation / file deletion / shell execution) → halt, show a
  visual diff, and await explicit user confirmation.

The classifier is heuristic (by action kind, plus destructive-command pattern
matching) and configurable. The guard fails closed: a ``CRITICAL`` action with
no confirmation handler is denied.
"""

from __future__ import annotations

import difflib
import inspect
import re
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    SAFE = "safe"
    MODERATE = "moderate"
    CRITICAL = "critical"


class Action(BaseModel):
    """A proposed action awaiting a safety verdict."""

    kind: str
    description: str = ""
    target: str | None = None
    command: str | None = None
    old: str | None = None  # prior content, for a diff
    new: str | None = None  # proposed content, for a diff
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    level: RiskLevel
    reason: str


class Decision(BaseModel):
    allowed: bool
    level: RiskLevel
    reason: str
    auto: bool = False
    diff: str | None = None


class ActionDenied(Exception):
    """A guarded action was not permitted to run."""


# Confirmation handlers may be sync or async.
ConfirmHandler = Callable[[Action, RiskAssessment], bool] | Callable[
    [Action, RiskAssessment], Awaitable[bool]
]


class RiskAssessor:
    """Classifies actions into risk levels."""

    SAFE_KINDS = {
        "read", "read_file", "list", "list_files", "compute", "math",
        "inspect", "search_memory", "noop",
    }
    MODERATE_KINDS = {
        "write", "write_file", "create", "create_file", "edit", "edit_file",
        "http_fetch", "api_fetch", "network", "search_web", "fetch",
    }
    CRITICAL_KINDS = {
        "delete", "delete_file", "remove", "rmtree", "shell", "bash", "exec",
        "command", "system", "overwrite", "deploy", "git_push",
    }
    DESTRUCTIVE_PATTERNS = [
        re.compile(p)
        for p in (
            r"rm\s+-rf?", r"\bsudo\b", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\s*\{",
            r">\s*/dev/sd", r"chmod\s+-R", r"git\s+push\b.*--force", r"\bshutdown\b",
        )
    ]

    def assess(self, action: Action) -> RiskAssessment:
        kind = action.kind.lower().strip()
        command = action.command or ""

        if kind in self.CRITICAL_KINDS:
            return RiskAssessment(level=RiskLevel.CRITICAL, reason=f"{kind!r} is a critical action")
        if command and any(p.search(command) for p in self.DESTRUCTIVE_PATTERNS):
            return RiskAssessment(
                level=RiskLevel.CRITICAL, reason="command matches a destructive pattern"
            )
        if kind in self.MODERATE_KINDS:
            return RiskAssessment(level=RiskLevel.MODERATE, reason=f"{kind!r} has side effects")
        if kind in self.SAFE_KINDS:
            return RiskAssessment(level=RiskLevel.SAFE, reason=f"{kind!r} is read-only or in-memory")
        return RiskAssessment(level=RiskLevel.MODERATE, reason=f"unclassified action {kind!r}")


def build_diff(action: Action) -> str | None:
    """Render a unified diff of ``old`` → ``new`` if either is present."""
    if action.old is None and action.new is None:
        return None
    old = (action.old or "").splitlines()
    new = (action.new or "").splitlines()
    label = action.target or action.kind
    diff = difflib.unified_diff(old, new, fromfile=f"a/{label}", tofile=f"b/{label}", lineterm="")
    return "\n".join(diff)


class SafetyGuard:
    """Reviews actions against a risk policy and a confirmation handler."""

    def __init__(
        self,
        assessor: RiskAssessor | None = None,
        confirm: ConfirmHandler | None = None,
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        self.assessor = assessor or RiskAssessor()
        self.confirm = confirm
        self.on_notice = on_notice or (lambda _msg: None)

    async def review(self, action: Action) -> Decision:
        assessment = self.assessor.assess(action)
        level, reason = assessment.level, assessment.reason

        if level is RiskLevel.SAFE:
            return Decision(allowed=True, level=level, reason=reason, auto=True)

        if level is RiskLevel.MODERATE:
            self.on_notice(f"[moderate] {action.kind}: {action.description or action.target or ''}")
            return Decision(allowed=True, level=level, reason=reason)

        # CRITICAL: halt, show the diff, and require explicit confirmation.
        diff = build_diff(action)
        self.on_notice(f"[CRITICAL] {action.kind}: {action.description or action.target or ''}")
        if diff:
            self.on_notice(diff)
        if self.confirm is None:
            return Decision(
                allowed=False,
                level=level,
                reason="critical action denied: no confirmation handler",
                diff=diff,
            )
        approved = await _call_confirm(self.confirm, action, assessment)
        return Decision(
            allowed=approved,
            level=level,
            reason="confirmed by user" if approved else "denied by user",
            diff=diff,
        )

    async def enforce(self, action: Action) -> Decision:
        """Like :meth:`review`, but raise :class:`ActionDenied` if not allowed."""
        decision = await self.review(action)
        if not decision.allowed:
            raise ActionDenied(decision.reason)
        return decision


async def _call_confirm(
    confirm: ConfirmHandler, action: Action, assessment: RiskAssessment
) -> bool:
    result = confirm(action, assessment)
    if inspect.isawaitable(result):
        result = await result
    return bool(result)
