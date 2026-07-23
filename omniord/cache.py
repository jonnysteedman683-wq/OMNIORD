"""Response caching layer for LLM calls and expensive computations.

Implements an in-memory LRU cache with optional persistence for frequently-used
queries, reducing redundant API calls and improving response latency.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def _hash_input(input_data: Any) -> str:
    """Generate a deterministic hash for cache keys.

    Handles dicts, lists, strings, and Pydantic models by serializing to JSON.
    """
    if isinstance(input_data, BaseModel):
        serialized = input_data.model_dump_json(sort_keys=True)
    elif isinstance(input_data, (dict, list)):
        serialized = json.dumps(input_data, sort_keys=True, default=str)
    else:
        serialized = str(input_data)

    return hashlib.sha256(serialized.encode()).hexdigest()


class CacheEntry(BaseModel):
    """A single cache entry with metadata."""

    key: str
    value: dict[str, Any] | str | None
    hits: int = 0
    created_at: float
    last_accessed_at: float


class ResponseCache:
    """In-memory LRU cache for LLM responses and computed results.

    Tracks cache hits/misses and supports optional TTL-based expiration.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float | None = None):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to store.
            ttl_seconds: Optional time-to-live for cached entries.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> dict[str, Any] | str | None:
        """Retrieve a value from the cache if it exists and is not expired."""
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check TTL expiration
        if self.ttl_seconds and (time.time() - entry.created_at) > self.ttl_seconds:
            del self.cache[key]
            return None

        # Update access tracking
        entry.hits += 1
        entry.last_accessed_at = time.time()
        return entry.value

    def put(self, key: str, value: dict[str, Any] | str) -> None:
        """Store a value in the cache, evicting old entries if needed."""
        if len(self.cache) >= self.max_size:
            # Evict least-recently-accessed entry
            lru_key = min(self.cache, key=lambda k: self.cache[k].last_accessed_at)
            del self.cache[lru_key]

        self.cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed_at=time.time(),
        )

    def get_or_compute(
        self,
        input_data: Any,
        compute_fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Get cached value or compute and cache it."""
        cache_key = _hash_input(input_data)
        cached = self.get(cache_key)

        if cached is not None:
            return cached

        # Compute and cache
        result = compute_fn(*args, **kwargs)
        self.put(cache_key, result)
        return result

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_hits = sum(e.hits for e in self.cache.values())
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "utilization_pct": (len(self.cache) / self.max_size * 100),
        }


# Process-wide response cache
_response_cache = ResponseCache(max_size=1000, ttl_seconds=3600)


def get_response_cache() -> ResponseCache:
    """Get the process-wide response cache."""
    return _response_cache


def reset_response_cache() -> None:
    """Clear the response cache (useful for tests)."""
    _response_cache.clear()
