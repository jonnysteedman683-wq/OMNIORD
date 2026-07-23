"""Self-evolving tool factory: AST safety, sandboxed execution, and a registry.

Generated code is statically screened by :mod:`ast_checker`, executed in an
isolated subprocess by :mod:`sandbox`, repaired by :mod:`factory`'s
self-healing loop, and persisted by :mod:`registry`.
"""

from __future__ import annotations

from omniord.tools.ast_checker import (
    AstSecurityError,
    SecurityPolicy,
    check_code,
    default_policy,
    is_safe,
)
from omniord.tools.factory import (
    BuildResult,
    CodeGenerator,
    RouterCodeGenerator,
    ToolFactory,
    ToolSpec,
)
from omniord.tools.registry import ToolRecord, ToolRegistry
from omniord.tools.sandbox import Sandbox, SandboxResult

__all__ = [
    "AstSecurityError",
    "BuildResult",
    "CodeGenerator",
    "RouterCodeGenerator",
    "Sandbox",
    "SandboxResult",
    "SecurityPolicy",
    "ToolFactory",
    "ToolRecord",
    "ToolRegistry",
    "ToolSpec",
    "check_code",
    "default_policy",
    "is_safe",
]
