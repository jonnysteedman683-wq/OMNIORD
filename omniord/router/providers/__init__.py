"""Concrete provider implementations for each backend."""

from __future__ import annotations

from omniord.config import CloudTierConfig
from omniord.router.base import LLMProvider
from omniord.router.providers.anthropic import AnthropicProvider
from omniord.router.providers.ollama import OllamaProvider
from omniord.router.providers.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "make_cloud_provider",
]


def make_cloud_provider(config: CloudTierConfig) -> LLMProvider | None:
    """Build the cloud provider selected in ``config``.

    Returns ``None`` when the selected provider has no API key configured, so
    the router can treat the cloud tier as unavailable instead of failing.
    """

    if not config.is_available:
        return None
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    return OpenAIProvider(config)
