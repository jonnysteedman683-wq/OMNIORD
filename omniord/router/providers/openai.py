"""OpenAI cloud provider (fallback tier)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence

import httpx

from omniord.config import CloudTierConfig
from omniord.router.base import (
    GenerationResult,
    LLMProvider,
    Message,
    ProviderError,
)


class OpenAIProvider(LLMProvider):
    """Talks to the OpenAI Chat Completions and Embeddings APIs."""

    name = "openai"
    tier = "cloud"

    def __init__(
        self,
        config: CloudTierConfig,
        *,
        base_url: str = "https://api.openai.com",
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.default_model = config.openai_model
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=config.request_timeout,
            headers={
                "authorization": f"Bearer {config.openai_api_key or ''}",
                "content-type": "application/json",
            },
        )

    @staticmethod
    def _dump(messages: Sequence[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(
        self, messages: Sequence[Message], *, model: str | None = None
    ) -> GenerationResult:
        model = model or self.default_model
        start = time.perf_counter()
        try:
            resp = await self._client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": self._dump(messages), "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"openai generate failed: {exc}") from exc
        latency = time.perf_counter() - start
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"openai returned an unexpected shape: {data!r}") from exc
        usage = data.get("usage") or {}
        return GenerationResult(
            text=text,
            provider=self.name,
            model=model,
            tier=self.tier,
            latency=latency,
            tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None
    ) -> AsyncIterator[str]:
        model = model or self.default_model
        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": model, "messages": self._dump(messages), "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    event = json.loads(payload)
                    delta = event["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise ProviderError(f"openai stream failed: {exc}") from exc

    async def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        model = model or "text-embedding-3-small"
        try:
            resp = await self._client.post(
                "/v1/embeddings", json={"model": model, "input": list(texts)}
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except (httpx.HTTPError, KeyError) as exc:
            raise ProviderError(f"openai embed failed: {exc}") from exc

    async def health_check(self) -> bool:
        if not self.config.openai_api_key:
            return False
        try:
            resp = await self._client.get("/v1/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
