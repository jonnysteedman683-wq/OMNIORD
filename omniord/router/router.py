"""The hybrid router: tiered policy + automatic local→cloud fallback.

Routing policy (from the architecture blueprint):

* **Tier 0 (local fast):** intent classification, DAG generation, parameter
  extraction — served by the local ``fast_model``.
* **Tier 1 (local code/reasoning):** code generation, tool synthesis,
  verification — served by the local ``code_model``.
* **Tier 2 (cloud fallback):** used when the local tier is unhealthy, errors,
  exceeds its latency limit, scores below the confidence threshold, or the
  caller forces it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Sequence
from enum import Enum
from typing import Literal

from omniord.config import OmniordSettings
from omniord.router.base import (
    GenerationResult,
    LLMProvider,
    Message,
    ProviderError,
    ProviderUnavailable,
)

ForceTier = Literal["local", "cloud"]
ConfidenceScorer = Callable[[GenerationResult], float]


class RouterError(Exception):
    """No viable tier could satisfy a request."""


class LowConfidence(ProviderError):
    """A local result scored below the configured confidence threshold."""


class TaskKind(str, Enum):
    """The kind of work being routed, which determines the target tier."""

    INTENT = "intent"      # tier 0
    DAG = "dag"            # tier 0
    EXTRACT = "extract"    # tier 0
    CODE = "code"          # tier 1
    REASON = "reason"      # tier 1
    VERIFY = "verify"      # tier 1


# Tier 0 vs tier 1 both run locally but on different models.
_TIER0 = {TaskKind.INTENT, TaskKind.DAG, TaskKind.EXTRACT}


class Router:
    """Routes generation across the local and cloud tiers with fallback."""

    def __init__(
        self,
        settings: OmniordSettings,
        *,
        local: LLMProvider,
        cloud: LLMProvider | None = None,
        health_ttl: float = 5.0,
    ):
        self.settings = settings
        self.local = local
        self.cloud = cloud
        self._health_ttl = health_ttl
        self._health_cache: tuple[float, bool] | None = None

    # ---------------- model selection ----------------

    def _local_model(self, task: TaskKind) -> str:
        return (
            self.settings.local.fast_model
            if task in _TIER0
            else self.settings.local.code_model
        )

    # ---------------- health (cached) ----------------

    async def _local_healthy(self) -> bool:
        now = time.monotonic()
        if self._health_cache is not None:
            checked_at, healthy = self._health_cache
            if now - checked_at < self._health_ttl:
                return healthy
        healthy = await self.local.health_check()
        self._health_cache = (now, healthy)
        return healthy

    # ---------------- generation ----------------

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        task: TaskKind = TaskKind.REASON,
        force: ForceTier | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
    ) -> GenerationResult:
        """Generate a completion, preferring the local tier per policy.

        ``force="local"`` disables cloud fallback (errors propagate);
        ``force="cloud"`` skips the local tier entirely.
        """
        go_local = force == "local" or (
            force is None and self.settings.prefer_local
        )

        if go_local:
            try:
                return await self._generate_local(messages, task, confidence_scorer)
            except (TimeoutError, ProviderError) as exc:  # LowConfidence is a ProviderError
                if force == "local":
                    raise RouterError(f"local tier failed and fallback is disabled: {exc}") from exc
                # fall through to the cloud tier
                fallback_reason: Exception | None = exc
        else:
            fallback_reason = None

        return await self._generate_cloud(messages, fell_back=fallback_reason is not None)

    async def _generate_local(
        self,
        messages: Sequence[Message],
        task: TaskKind,
        confidence_scorer: ConfidenceScorer | None,
    ) -> GenerationResult:
        if not await self._local_healthy():
            raise ProviderUnavailable("local tier failed its health check")
        result = await asyncio.wait_for(
            self.local.generate(messages, model=self._local_model(task)),
            timeout=self.settings.local.latency_limit,
        )
        if confidence_scorer is not None:
            score = confidence_scorer(result)
            if score < self.settings.local.confidence_threshold:
                raise LowConfidence(
                    f"local confidence {score:.2f} below threshold "
                    f"{self.settings.local.confidence_threshold:.2f}"
                )
        return result

    async def _generate_cloud(
        self, messages: Sequence[Message], *, fell_back: bool
    ) -> GenerationResult:
        if self.cloud is None:
            raise RouterError("cloud tier is unavailable (no provider or API key configured)")
        result = await self.cloud.generate(messages, model=self.settings.cloud.active_model)
        result.fell_back = fell_back
        return result

    # ---------------- streaming ----------------

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        task: TaskKind = TaskKind.REASON,
        force: ForceTier | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion. Chooses a tier up front; it does not switch
        tiers mid-stream once tokens have started arriving."""
        go_local = force == "local" or (force is None and self.settings.prefer_local)
        provider: LLMProvider
        model: str | None
        if go_local and await self._local_healthy():
            provider, model = self.local, self._local_model(task)
        elif force == "local":
            raise RouterError("local tier is unhealthy and fallback is disabled")
        elif self.cloud is not None:
            provider, model = self.cloud, self.settings.cloud.active_model
        else:
            raise RouterError("no viable tier available for streaming")
        async for piece in provider.stream(messages, model=model):
            yield piece

    # ---------------- embeddings ----------------

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed via the local tier, falling back to the cloud tier on failure."""
        try:
            if await self._local_healthy():
                return await self.local.embed(texts)
        except ProviderError:
            pass
        if self.cloud is not None:
            return await self.cloud.embed(texts)
        raise RouterError("no tier could produce embeddings")

    async def aclose(self) -> None:
        await self.local.aclose()
        if self.cloud is not None:
            await self.cloud.aclose()
