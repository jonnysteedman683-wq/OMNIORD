"""HTTP connection pooling for efficient reuse across LLM provider calls.

Manages a process-wide httpx.AsyncClient with connection pooling, retry logic,
and automatic resource cleanup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx


class HTTPConnectionPool:
    """Singleton HTTP client pool with automatic connection reuse."""

    _instance: HTTPConnectionPool | None = None
    _client: httpx.AsyncClient | None = None

    def __new__(cls) -> HTTPConnectionPool:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_client(self, timeout: float = 120.0) -> httpx.AsyncClient:
        """Get or create the shared async HTTP client.

        Uses connection pooling for performance. The client is created lazily
        and reused across all calls.
        """
        if self._client is None:
            limits = httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            )
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def reset(self) -> None:
        """Reset the client pool (useful for tests)."""
        await self.close()


# Global HTTP pool instance
_http_pool = HTTPConnectionPool()


async def get_http_client(timeout: float = 120.0) -> httpx.AsyncClient:
    """Get the shared HTTP client for LLM calls."""
    return await _http_pool.get_client(timeout)


async def close_http_pool() -> None:
    """Cleanup HTTP resources at shutdown."""
    await _http_pool.close()


@asynccontextmanager
async def http_session(timeout: float = 120.0) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Context manager for using the pooled HTTP client.

    Example:
        async with http_session() as client:
            response = await client.get("http://localhost:11434/api/tags")
    """
    client = await get_http_client(timeout)
    try:
        yield client
    except Exception:
        # Don't close the pooled client on error; it can be reused
        raise
