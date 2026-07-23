"""Local Ollama provider (the default, local-first tier)."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator, Sequence

import httpx

from omniord.config import LocalTierConfig
from omniord.router.base import (
    GenerationResult,
    LLMProvider,
    Message,
    ProviderError,
)


class OllamaProvider(LLMProvider):
    """Talks to an Ollama server over its native HTTP API."""

    name = "ollama"
    tier = "local"

    def __init__(self, config: LocalTierConfig, *, client: httpx.AsyncClient | None = None):
        self.config = config
        self.default_model = config.fast_model
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url, timeout=config.request_timeout
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
                "/api/chat",
                json={"model": model, "messages": self._dump(messages), "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama generate failed: {exc}") from exc
        latency = time.perf_counter() - start
        content = (data.get("message") or {}).get("content")
        if content is None:
            raise ProviderError(f"ollama returned no message content: {data!r}")
        return GenerationResult(
            text=content,
            provider=self.name,
            model=model,
            tier=self.tier,
            latency=latency,
            tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason") or ("stop" if data.get("done") else None),
        )

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None
    ) -> AsyncIterator[str]:
        model = model or self.default_model
        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json={"model": model, "messages": self._dump(messages), "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = (chunk.get("message") or {}).get("content")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama stream failed: {exc}") from exc

    async def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        model = model or self.default_model
        vectors: list[list[float]] = []
        try:
            for text in texts:
                resp = await self._client.post(
                    "/api/embeddings", json={"model": model, "prompt": text}
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        except (httpx.HTTPError, KeyError) as exc:
            raise ProviderError(f"ollama embed failed: {exc}") from exc
        return vectors

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
