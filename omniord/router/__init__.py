"""Hybrid edge/cloud LLM router.

Omniord prefers a local tier (Ollama) and escalates to a cloud tier
(Anthropic / OpenAI) only when the local tier fails a health check, errors, or
exceeds its latency limit. The :class:`~omniord.router.router.Router` applies
that policy; providers behind it share the interface in
:mod:`omniord.router.base`.
"""

from __future__ import annotations

from omniord.router.base import (
    GenerationResult,
    LLMProvider,
    Message,
    ProviderError,
    ProviderUnavailable,
    Role,
)
from omniord.router.router import Router, RouterError, TaskKind

__all__ = [
    "GenerationResult",
    "LLMProvider",
    "Message",
    "ProviderError",
    "ProviderUnavailable",
    "Role",
    "Router",
    "RouterError",
    "TaskKind",
]
