"""Static AST security analysis for dynamically generated code.

Before any generated code is executed, it is parsed with the built-in ``ast``
module and screened against a :class:`SecurityPolicy`. The checker blocks unsafe
imports (``os``, ``subprocess``, ``socket``, …), code-injection builtins
(``eval``, ``exec``, ``compile``, ``__import__``), known-dangerous calls
(``os.system``, ``subprocess.Popen``, ``shutil.rmtree``, …), and dunder-attribute
access used for sandbox escapes (``__globals__``, ``__subclasses__``, …).

This is a *static* gate, not a sandbox — it runs before, and in addition to,
:mod:`omniord.tools.sandbox`.
"""

from __future__ import annotations

import ast

from pydantic import BaseModel, Field


class AstSecurityError(Exception):
    """Generated code failed static security screening."""


class SecurityPolicy(BaseModel):
    """What the AST checker forbids. Names are matched by their root module or
    fully-dotted call path."""

    forbidden_imports: set[str] = Field(
        default_factory=lambda: {
            "os",
            "sys",
            "subprocess",
            "socket",
            "shutil",
            "ctypes",
            "importlib",
            "multiprocessing",
            "resource",
            "signal",
            "pty",
            "fcntl",
            "mmap",
        }
    )
    forbidden_calls: set[str] = Field(
        default_factory=lambda: {
            "eval",
            "exec",
            "compile",
            "__import__",
            "os.system",
            "os.popen",
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.run",
            "subprocess.check_output",
            "shutil.rmtree",
            "socket.socket",
        }
    )
    forbid_dunder_access: bool = True


def default_policy() -> SecurityPolicy:
    return SecurityPolicy()


def _dotted_name(node: ast.expr) -> str | None:
    """Return the dotted path for a Name/Attribute chain, else None."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self, policy: SecurityPolicy) -> None:
        self.policy = policy
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.violations.append(f"line {line}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in self.policy.forbidden_imports:
                self._flag(node, f"forbidden import {alias.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in self.policy.forbidden_imports:
            self._flag(node, f"forbidden import from {node.module!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.policy.forbidden_calls:
            self._flag(node, f"forbidden call {node.func.id!r}")
        else:
            dotted = _dotted_name(node.func)
            if dotted and dotted in self.policy.forbidden_calls:
                self._flag(node, f"forbidden call {dotted!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            self.policy.forbid_dunder_access
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            self._flag(node, f"forbidden dunder attribute access {node.attr!r}")
        self.generic_visit(node)


def check_code(source: str, policy: SecurityPolicy | None = None) -> list[str]:
    """Return a list of security violations (empty means the code passed).

    A syntax error is reported as a single violation rather than raised, so the
    tool factory can route it back through its repair loop.
    """
    policy = policy or default_policy()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    scanner = _Scanner(policy)
    scanner.visit(tree)
    return scanner.violations


def is_safe(source: str, policy: SecurityPolicy | None = None) -> bool:
    return not check_code(source, policy)


def assert_safe(source: str, policy: SecurityPolicy | None = None) -> None:
    """Raise :class:`AstSecurityError` if ``source`` violates the policy."""
    violations = check_code(source, policy)
    if violations:
        raise AstSecurityError("; ".join(violations))
