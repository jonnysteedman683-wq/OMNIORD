"""Lazy loading utilities for deferred provider SDK initialization.

Reduces startup time by loading heavy dependencies (Anthropic, OpenAI SDKs) only
when actually needed, not at import time.
"""

from __future__ import annotations

from typing import Any


class LazyLoader:
    """Lazy-load a module on first access."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module: Any = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            import importlib

            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)


# Lazily load heavy provider SDKs
anthropic = LazyLoader("anthropic")
openai = LazyLoader("openai")


def lazy_import(module_name: str) -> Any:
    """Dynamically import a module on first use.

    Example:
        httpx = lazy_import("httpx")
        client = httpx.AsyncClient()
    """
    return LazyLoader(module_name)
