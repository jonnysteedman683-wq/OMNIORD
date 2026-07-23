"""Phase 2 tests: provider connectors and the hybrid router's fallback policy."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Sequence

import httpx
import pytest

from omniord.config import CloudTierConfig, LocalTierConfig, OmniordSettings
from omniord.router.base import (
    GenerationResult,
    LLMProvider,
    Message,
    ProviderError,
)
from omniord.router.providers.ollama import OllamaProvider
from omniord.router.providers.openai import OpenAIProvider
from omniord.router.router import Router, RouterError, TaskKind

USER = [Message(role="user", content="hello")]


# --------------------------------------------------------------------------- #
# Provider connectors (HTTP mocked, no network)
# --------------------------------------------------------------------------- #


async def test_ollama_generate_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "message": {"content": "hi there"},
                "eval_count": 5,
                "done": True,
                "done_reason": "stop",
            },
        )

    client = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    provider = OllamaProvider(LocalTierConfig(), client=client)
    result = await provider.generate(USER)
    assert result.text == "hi there"
    assert result.tier == "local"
    assert result.tokens == 5
    assert result.finish_reason == "stop"
    await provider.aclose()


async def test_ollama_health_and_embed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/embeddings":
            return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
        return httpx.Response(404)

    client = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    provider = OllamaProvider(LocalTierConfig(), client=client)
    assert await provider.health_check() is True
    vectors = await provider.embed(["a", "b"])
    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    await provider.aclose()


async def test_ollama_generate_raises_provider_error_on_500() -> None:
    client = httpx.AsyncClient(
        base_url="http://ollama",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    provider = OllamaProvider(LocalTierConfig(), client=client)
    with pytest.raises(ProviderError):
        await provider.generate(USER)
    assert await provider.health_check() is False
    await provider.aclose()


async def test_openai_generate_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "cloud says hi"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 7},
            },
        )

    client = httpx.AsyncClient(base_url="http://openai", transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(CloudTierConfig(openai_api_key="x"), client=client)
    result = await provider.generate(USER)
    assert result.text == "cloud says hi"
    assert result.tier == "cloud"
    assert result.tokens == 7
    await provider.aclose()


# --------------------------------------------------------------------------- #
# Router policy (in-memory fake providers)
# --------------------------------------------------------------------------- #


class FakeProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        tier: str,
        *,
        healthy: bool = True,
        text: str = "ok",
        error: Exception | None = None,
        delay: float = 0.0,
    ):
        self.name = name
        self.tier = tier  # type: ignore[assignment]
        self.healthy = healthy
        self.text = text
        self.error = error
        self.delay = delay
        self.models_seen: list[str | None] = []
        self.closed = False

    async def generate(self, messages: Sequence[Message], *, model: str | None = None) -> GenerationResult:
        self.models_seen.append(model)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return GenerationResult(text=self.text, provider=self.name, model=model or "?", tier=self.tier)

    async def stream(self, messages: Sequence[Message], *, model: str | None = None) -> AsyncIterator[str]:
        for word in self.text.split():
            yield word

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        if self.error:
            raise self.error
        return [[1.0] for _ in texts]

    async def health_check(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:
        self.closed = True


def _settings(**over: object) -> OmniordSettings:
    settings = OmniordSettings(_env_file=None)
    settings.cloud = CloudTierConfig(provider="anthropic", anthropic_api_key="k")
    for key, value in over.items():
        setattr(settings, key, value)
    return settings


async def test_prefers_local_when_healthy() -> None:
    local = FakeProvider("ollama", "local", text="local answer")
    cloud = FakeProvider("anthropic", "cloud", text="cloud answer")
    router = Router(_settings(), local=local, cloud=cloud)
    result = await router.generate(USER, task=TaskKind.REASON)
    assert result.text == "local answer"
    assert result.tier == "local"
    assert result.fell_back is False
    assert cloud.models_seen == []


async def test_falls_back_when_local_unhealthy() -> None:
    local = FakeProvider("ollama", "local", healthy=False)
    cloud = FakeProvider("anthropic", "cloud", text="cloud answer")
    router = Router(_settings(), local=local, cloud=cloud)
    result = await router.generate(USER)
    assert result.text == "cloud answer"
    assert result.tier == "cloud"
    assert result.fell_back is True
    assert local.models_seen == []  # never even attempted the call


async def test_falls_back_when_local_errors() -> None:
    local = FakeProvider("ollama", "local", error=ProviderError("kaboom"))
    cloud = FakeProvider("anthropic", "cloud", text="cloud answer")
    router = Router(_settings(), local=local, cloud=cloud)
    result = await router.generate(USER)
    assert result.tier == "cloud"
    assert result.fell_back is True


async def test_falls_back_on_latency_timeout() -> None:
    settings = _settings()
    settings.local.latency_limit = 0.02
    local = FakeProvider("ollama", "local", delay=0.5)
    cloud = FakeProvider("anthropic", "cloud", text="cloud answer")
    router = Router(settings, local=local, cloud=cloud)
    result = await router.generate(USER)
    assert result.tier == "cloud"
    assert result.fell_back is True


async def test_confidence_threshold_triggers_fallback() -> None:
    settings = _settings()
    settings.local.confidence_threshold = 0.9
    local = FakeProvider("ollama", "local", text="meh")
    cloud = FakeProvider("anthropic", "cloud", text="cloud answer")
    router = Router(settings, local=local, cloud=cloud)
    result = await router.generate(USER, confidence_scorer=lambda r: 0.1)
    assert result.tier == "cloud"
    assert result.fell_back is True


async def test_force_local_disables_fallback() -> None:
    local = FakeProvider("ollama", "local", healthy=False)
    cloud = FakeProvider("anthropic", "cloud")
    router = Router(_settings(), local=local, cloud=cloud)
    with pytest.raises(RouterError):
        await router.generate(USER, force="local")
    assert cloud.models_seen == []


async def test_force_cloud_skips_local() -> None:
    local = FakeProvider("ollama", "local")
    cloud = FakeProvider("anthropic", "cloud", text="cloud answer")
    router = Router(_settings(), local=local, cloud=cloud)
    result = await router.generate(USER, force="cloud")
    assert result.tier == "cloud"
    assert result.fell_back is False  # a direct route, not a fallback
    assert local.models_seen == []


async def test_tier0_and_tier1_select_different_models() -> None:
    local = FakeProvider("ollama", "local")
    router = Router(_settings(), local=local, cloud=None)
    await router.generate(USER, task=TaskKind.INTENT)   # tier 0 -> fast_model
    await router.generate(USER, task=TaskKind.CODE)      # tier 1 -> code_model
    assert local.models_seen == ["llama3.1", "qwen2.5-coder"]


async def test_router_error_when_local_fails_and_no_cloud() -> None:
    local = FakeProvider("ollama", "local", healthy=False)
    router = Router(_settings(), local=local, cloud=None)
    with pytest.raises(RouterError):
        await router.generate(USER)


async def test_stream_uses_local_when_healthy() -> None:
    local = FakeProvider("ollama", "local", text="one two three")
    cloud = FakeProvider("anthropic", "cloud", text="a b c")
    router = Router(_settings(), local=local, cloud=cloud)
    pieces = [chunk async for chunk in router.stream(USER)]
    assert pieces == ["one", "two", "three"]


async def test_embed_falls_back_to_cloud() -> None:
    local = FakeProvider("ollama", "local", error=ProviderError("no embed"))
    cloud = FakeProvider("anthropic", "cloud")
    router = Router(_settings(), local=local, cloud=cloud)
    vectors = await router.embed(["x"])
    assert vectors == [[1.0]]


async def test_aclose_closes_both_tiers() -> None:
    local = FakeProvider("ollama", "local")
    cloud = FakeProvider("anthropic", "cloud")
    router = Router(_settings(), local=local, cloud=cloud)
    await router.aclose()
    assert local.closed and cloud.closed
