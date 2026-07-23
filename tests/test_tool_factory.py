"""Phase 4 tests: AST checker, sandbox, registry, and the self-healing factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from omniord.config import OmniordSettings
from omniord.router.base import GenerationResult, LLMProvider, Message
from omniord.router.router import Router
from omniord.tools.ast_checker import (
    AstSecurityError,
    assert_safe,
    check_code,
    is_safe,
)
from omniord.tools.factory import (
    RouterCodeGenerator,
    ToolFactory,
    ToolSpec,
    extract_code,
)
from omniord.tools.registry import ToolRegistry
from omniord.tools.sandbox import Sandbox

# --------------------------------------------------------------------------- #
# AST checker
# --------------------------------------------------------------------------- #


def test_safe_code_passes() -> None:
    assert is_safe("import math\n\ndef area(r):\n    return math.pi * r * r\n")


def test_forbidden_import_flagged() -> None:
    violations = check_code("import os\n")
    assert violations and "forbidden import" in violations[0]


def test_forbidden_import_from_flagged() -> None:
    assert not is_safe("from subprocess import Popen\n")


def test_eval_and_exec_flagged() -> None:
    assert not is_safe("x = eval('1+1')\n")
    assert not is_safe("exec('y = 2')\n")


def test_dunder_access_flagged() -> None:
    assert not is_safe("def f(x):\n    return x.__globals__\n")


def test_syntax_error_reported_not_raised() -> None:
    violations = check_code("def broken(:\n")
    assert violations and "syntax error" in violations[0]


def test_assert_safe_raises_with_details() -> None:
    with pytest.raises(AstSecurityError) as exc:
        assert_safe("import socket\n")
    assert "forbidden import" in str(exc.value)


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #


async def test_sandbox_runs_code_and_captures_stdout() -> None:
    result = await Sandbox().run("print('hello sandbox')")
    assert result.ok
    assert "hello sandbox" in result.stdout
    assert result.returncode == 0


async def test_sandbox_reports_runtime_error() -> None:
    result = await Sandbox().run("raise ValueError('nope')")
    assert not result.ok
    assert result.returncode != 0
    assert "ValueError" in result.stderr


async def test_sandbox_enforces_timeout() -> None:
    result = await Sandbox(timeout=0.3).run("while True:\n    pass\n")
    assert result.timed_out
    assert not result.ok


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "tools")
    registry.register(
        "adder",
        "def add(a, b):\n    return a + b\n",
        schema={"type": "object"},
        description="adds two numbers",
    )
    record = registry.get("adder")
    assert record is not None
    assert record.description == "adds two numbers"
    assert record.schema_ == {"type": "object"}
    assert "def add" in registry.load_source("adder")
    assert [r.name for r in registry.list()] == ["adder"]

    # A fresh instance reads the persisted index.
    reopened = ToolRegistry(tmp_path / "tools")
    assert reopened.get("adder") is not None

    assert reopened.remove("adder") is True
    assert reopened.get("adder") is None


def test_registry_rejects_bad_names(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "tools")
    with pytest.raises(ValueError):
        registry.register("bad name!", "x = 1\n")


# --------------------------------------------------------------------------- #
# Factory self-healing loop
# --------------------------------------------------------------------------- #

_GOOD = "def add(a, b):\n    return a + b\n"
_WRONG = "def add(a, b):\n    return a - b\n"       # runtime: fails validation
_UNSAFE = "import os\ndef add(a, b):\n    return a + b\n"  # AST violation
_SPEC = ToolSpec(
    name="add",
    description="add two numbers",
    validation_code="assert add(2, 3) == 5\n",
)


class FakeGenerator:
    """Serves a fixed sequence of code: generate() yields the first item,
    each repair() yields the next."""

    def __init__(self, codes: list[str]) -> None:
        self.codes = codes
        self.idx = 0
        self.repair_errors: list[str] = []

    async def generate(self, spec: ToolSpec) -> str:
        self.idx = 0
        return self.codes[0]

    async def repair(self, spec: ToolSpec, code: str, error: str) -> str:
        self.repair_errors.append(error)
        self.idx += 1
        return self.codes[self.idx]


async def test_build_succeeds_first_try(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "tools")
    factory = ToolFactory(FakeGenerator([_GOOD]), registry)
    result = await factory.build(_SPEC)
    assert result.success is True
    assert result.attempts == 1
    assert result.registered is True
    assert registry.get("add") is not None


async def test_build_repairs_runtime_failure(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "tools")
    generator = FakeGenerator([_WRONG, _GOOD])
    factory = ToolFactory(generator, registry)
    result = await factory.build(_SPEC)
    assert result.success is True
    assert result.attempts == 2
    assert len(generator.repair_errors) == 1
    assert registry.get("add") is not None


async def test_build_repairs_ast_violation(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "tools")
    generator = FakeGenerator([_UNSAFE, _GOOD])
    factory = ToolFactory(generator, registry)
    result = await factory.build(_SPEC)
    assert result.success is True
    assert "AST security violations" in generator.repair_errors[0]


async def test_build_fails_after_exhausting_retries(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path / "tools")
    generator = FakeGenerator([_WRONG] * 5)
    factory = ToolFactory(generator, registry, max_retries=3)
    result = await factory.build(_SPEC)
    assert result.success is False
    assert result.registered is False
    assert result.attempts == 4  # initial + 3 repairs
    assert registry.get("add") is None


# --------------------------------------------------------------------------- #
# extract_code + RouterCodeGenerator (Phase 2 ↔ 4 wiring)
# --------------------------------------------------------------------------- #


def test_extract_code_strips_fences() -> None:
    assert extract_code("```python\nx = 1\n```") == "x = 1"
    assert extract_code("no fences here") == "no fences here"


class _CodeProvider(LLMProvider):
    name = "fake"
    tier = "local"

    async def generate(self, messages: Sequence[Message], *, model: str | None = None) -> GenerationResult:
        return GenerationResult(
            text="```python\ndef add(a, b):\n    return a + b\n```",
            provider=self.name,
            model=model or "m",
            tier=self.tier,
        )

    async def stream(self, messages: Sequence[Message], *, model: str | None = None) -> AsyncIterator[str]:
        yield ""

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        return []

    async def health_check(self) -> bool:
        return True


async def test_router_code_generator_returns_unfenced_code() -> None:
    router = Router(OmniordSettings(_env_file=None), local=_CodeProvider(), cloud=None)
    generator = RouterCodeGenerator(router)
    code = await generator.generate(ToolSpec(name="add"))
    assert code == "def add(a, b):\n    return a + b"
