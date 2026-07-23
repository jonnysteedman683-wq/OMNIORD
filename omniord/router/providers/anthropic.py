"""Anthropic cloud provider (fallback tier)."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator, Sequence

import httpx

from omniord.config import CloudTierConfig
from omniord.router.base import (
    GenerationResult,
    LLMProvider,
    Message,
    ProviderError,
)

_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    """Talks to the Anthropic Messages API."""

    name = "anthropic"
    tier = "cloud"

    def __init__(
        self,
        config: CloudTierConfig,
        *,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.default_model = config.anthropic_model
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=config.request_timeout,
            headers={
                "x-api-key": config.anthropic_api_key or "",
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            },
        )

    @staticmethod
    def _split(messages: Sequence[Message]) -> tuple[str | None, list[dict]]:
        """Anthropic takes the system prompt separately from the turn list."""
        system_parts = [m.content for m in messages if m.role == "system"]
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, turns

    def _payload(self, messages: Sequence[Message], model: str, stream: bool) -> dict:
        system, turns = self._split(messages)
        payload: dict = {
            "model": model,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "messages": turns,
            "stream": stream,
        }
        if system is not None:
            payload["system"] = system
        return payload

    async def generate(
        self, messages: Sequence[Message], *, model: str | None = None
    ) -> GenerationResult:
        model = model or self.default_model
        start = time.perf_counter()
        try:
            resp = await self._client.post("/v1/messages", json=self._payload(messages, model, False))
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"anthropic generate failed: {exc}") from exc
        latency = time.perf_counter() - start
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        return GenerationResult(
            text=text,
            provider=self.name,
            model=model,
            tier=self.tier,
            latency=latency,
            tokens=usage.get("output_tokens"),
            finish_reason=data.get("stop_reason"),
        )

    async def stream(
        self, messages: Sequence[Message], *, model: str | None = None
    ) -> AsyncIterator[str]:
        model = model or self.default_model
        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=self._payload(messages, model, True)
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[len("data:") :].strip())
                    if event.get("type") == "content_block_delta":
                        piece = (event.get("delta") or {}).get("text")
                        if piece:
                            yield piece
        except httpx.HTTPError as exc:
            raise ProviderError(f"anthropic stream failed: {exc}") from exc

    async def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        raise ProviderError("anthropic does not provide an embeddings API")

    async def health_check(self) -> bool:
        if not self.config.anthropic_api_key:
            return False
        try:
            resp = await self._client.get("/v1/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
