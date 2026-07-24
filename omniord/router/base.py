"""Provider interface and message/response schemas shared across the router.

Every backend — local Ollama or a cloud API — implements :class:`LLMProvider`,
exposing a unified async surface (:meth:`~LLMProvider.generate`,
:meth:`~LLMProvider.stream`, :meth:`~LLMProvider.embed`,
:meth:`~LLMProvider.health_check`). Messages and results are Pydantic models so
they are validated at the boundary and passed inward as typed objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]
Tier = Literal["local", "cloud"]


class ProviderError(Exception):
    """A provider failed to complete a request (bad response, API error)."""


class ProviderUnavailable(ProviderError):
    """A provider is unreachable or failed its health check."""


class Message(BaseModel):
    """A single chat message."""

    role: Role
    content: str


class GenerationResult(BaseModel):
    """The outcome of a (non-streaming) generation call."""

    text: str
    provider: str
    model: str
    tier: Tier
    latency: float = Field(default=0.0, description="Wall-clock seconds for the call")
    tokens: int | None = None
    finish_reason: str | None = None
    # Set by the Router when this result came from an escalation to the cloud
    # tier after the local tier was skipped or failed.
    fell_back: bool = False


class LLMProvider(ABC):
    """Abstract base for every model backend."""

    #: Human-readable provider name, e.g. "ollama", "anthropic".
    name: str
    #: Which tier this provider serves.
    tier: Tier

    @abstractmethod
    async def generate(
        self, messages: Sequence[Message], *, model: str | None = None
    ) -> GenerationResult:
        """Return a full completion for ``messages``."""

    @abstractmethod
    def stream(
        self, messages: Sequence[Message], *, model: str | None = None
    ) -> AsyncIterator[str]:
        """Yield completion text chunks as they arrive.

        Implemented as an async generator, so it is defined without ``async``
        here but returns an async iterator at call time.
        """

    @abstractmethod
    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Return an embedding vector for each input text."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and usable right now."""

    async def aclose(self) -> None:
        """Release any underlying resources (HTTP clients, sockets)."""
        return None
