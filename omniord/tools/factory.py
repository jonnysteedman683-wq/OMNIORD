"""The self-evolving tool factory with a self-healing repair loop.

``ToolFactory.build`` runs the cycle from the blueprint:

    generate code → static AST analysis → sandbox test execution
        → repair on error → save to registry

A failing static check or a failing sandbox run is fed back to the code
generator for repair, up to ``max_retries`` times, before the build is reported
as failed. The generator is injected (a ``CodeGenerator``), so the factory is
independent of any specific LLM; :class:`RouterCodeGenerator` adapts the
Phase-2 :class:`~omniord.router.router.Router` to that interface.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from omniord.router.base import Message
from omniord.router.router import Router, TaskKind
from omniord.tools.ast_checker import SecurityPolicy, check_code
from omniord.tools.registry import ToolRegistry
from omniord.tools.sandbox import Sandbox


class ToolSpec(BaseModel):
    """A description of the tool to synthesize."""

    name: str
    description: str = ""
    instructions: str = ""
    input_schema: dict = Field(default_factory=dict)
    # Trusted Python appended after the generated code and run in the sandbox to
    # prove the tool works (e.g. ``assert add(2, 3) == 5``).
    validation_code: str = ""


class BuildResult(BaseModel):
    success: bool
    name: str
    attempts: int
    source: str | None = None
    error: str | None = None
    registered: bool = False


@runtime_checkable
class CodeGenerator(Protocol):
    """Produces and repairs tool source code."""

    async def generate(self, spec: ToolSpec) -> str: ...

    async def repair(self, spec: ToolSpec, code: str, error: str) -> str: ...


class ToolFactory:
    """Generates, validates, sandbox-tests, repairs, and registers tools."""

    def __init__(
        self,
        generator: CodeGenerator,
        registry: ToolRegistry,
        *,
        sandbox: Sandbox | None = None,
        policy: SecurityPolicy | None = None,
        max_retries: int = 3,
    ) -> None:
        self.generator = generator
        self.registry = registry
        self.sandbox = sandbox or Sandbox()
        self.policy = policy
        self.max_retries = max_retries

    async def build(self, spec: ToolSpec) -> BuildResult:
        code = await self.generator.generate(spec)
        last_error: str | None = None
        attempt = 0
        for attempt in range(1, self.max_retries + 2):
            violations = check_code(code, self.policy)
            if violations:
                last_error = "AST security violations: " + "; ".join(violations)
            else:
                result = await self.sandbox.run(code + "\n" + spec.validation_code)
                if result.ok:
                    self.registry.register(
                        spec.name, code, spec.input_schema, spec.description
                    )
                    return BuildResult(
                        success=True,
                        name=spec.name,
                        attempts=attempt,
                        source=code,
                        registered=True,
                    )
                last_error = (
                    "execution timed out"
                    if result.timed_out
                    else (result.stderr.strip()[:1000] or "tool failed its validation check")
                )

            if attempt <= self.max_retries:
                code = await self.generator.repair(spec, code, last_error)

        return BuildResult(
            success=False,
            name=spec.name,
            attempts=attempt,
            source=code,
            error=last_error,
            registered=False,
        )


_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the code out of a model response, stripping Markdown fences."""
    match = _FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


class RouterCodeGenerator:
    """Adapts the hybrid :class:`Router` to the :class:`CodeGenerator` interface."""

    _SYSTEM = (
        "You are a Python tool-writing assistant. Return only a single, "
        "self-contained Python module implementing the requested tool. Do not "
        "use os, subprocess, socket, or other system-access modules."
    )

    def __init__(self, router: Router) -> None:
        self.router = router

    async def generate(self, spec: ToolSpec) -> str:
        prompt = (
            f"Write a Python tool named {spec.name!r}.\n"
            f"Description: {spec.description}\n"
            f"Instructions: {spec.instructions}\n"
            f"Input schema (JSON): {spec.input_schema}"
        )
        result = await self.router.generate(
            [Message(role="system", content=self._SYSTEM), Message(role="user", content=prompt)],
            task=TaskKind.CODE,
        )
        return extract_code(result.text)

    async def repair(self, spec: ToolSpec, code: str, error: str) -> str:
        prompt = (
            f"The tool {spec.name!r} failed. Fix it and return the full corrected "
            f"module.\n\nError:\n{error}\n\nCurrent code:\n{code}"
        )
        result = await self.router.generate(
            [Message(role="system", content=self._SYSTEM), Message(role="user", content=prompt)],
            task=TaskKind.CODE,
        )
        return extract_code(result.text)
