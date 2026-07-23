"""Configuration for Omniord, resolved from environment variables and .env.

Settings are grouped into two tiers that mirror the hybrid engine: a local
tier (Ollama) that Omniord prefers, and a cloud tier (Anthropic / OpenAI) it
escalates to only on fallback. Everything is a Pydantic model, so values are
validated at the boundary and passed inward as typed objects.

Environment variables use the ``OMNIORD_`` prefix and ``__`` to descend into a
nested group, e.g.::

    OMNIORD_LOCAL__FAST_MODEL=llama3.2
    OMNIORD_CLOUD__PROVIDER=openai
    OMNIORD_CLOUD__ANTHROPIC_API_KEY=sk-ant-...
    OMNIORD_MAX_RETRIES=3
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CloudProvider = Literal["anthropic", "openai"]


class LocalTierConfig(BaseModel):
    """Local-first tier, backed by an Ollama endpoint.

    ``fast_model`` handles Tier 0 work (intent classification, DAG generation,
    parameter extraction); ``code_model`` handles Tier 1 work (code generation,
    tool synthesis, verification).
    """

    base_url: str = "http://localhost:11434"
    fast_model: str = "llama3.1"
    code_model: str = "qwen2.5-coder"
    request_timeout: float = 120.0
    # Escalation triggers: fall back to the cloud tier when a local response
    # scores below this confidence or takes longer than this many seconds.
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    latency_limit: float = Field(default=30.0, gt=0.0)


class CloudTierConfig(BaseModel):
    """Cloud fallback tier (Anthropic / OpenAI).

    API keys default to ``None``; the router treats a tier whose selected
    provider has no key as unavailable rather than raising at import time.
    """

    provider: CloudProvider = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    request_timeout: float = 120.0

    @property
    def active_model(self) -> str:
        return self.anthropic_model if self.provider == "anthropic" else self.openai_model

    @property
    def active_key(self) -> str | None:
        return self.anthropic_api_key if self.provider == "anthropic" else self.openai_api_key

    @property
    def is_available(self) -> bool:
        """True when the selected provider has a key configured."""
        return bool(self.active_key)


class OmniordSettings(BaseSettings):
    """Top-level Omniord configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OMNIORD_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    local: LocalTierConfig = Field(default_factory=LocalTierConfig)
    cloud: CloudTierConfig = Field(default_factory=CloudTierConfig)

    # Directory Omniord's tools and sandbox operate in.
    workspace: Path = Field(default_factory=Path.cwd)
    # Prefer the local tier for everything it can handle before escalating.
    prefer_local: bool = True
    # Bound on the self-healing reflection loop's auto-retry attempts.
    max_retries: int = Field(default=3, ge=0)
    # Hard timeout (seconds) for code run in the isolated sandbox.
    sandbox_timeout: float = Field(default=30.0, gt=0.0)


@lru_cache
def get_settings() -> OmniordSettings:
    """Return the process-wide settings, loaded once and cached.

    Call ``get_settings.cache_clear()`` in tests that mutate the environment.
    """

    return OmniordSettings()
