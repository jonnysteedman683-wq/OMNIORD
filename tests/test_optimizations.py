"""Integration tests for performance optimizations.

Validates that HTTP pooling, caching, metrics tracking, and lazy loading
work correctly together.
"""

import pytest
import time
from omniord.http_pool import get_http_client, close_http_pool
from omniord.cache import get_response_cache, reset_response_cache
from omniord.metrics import get_metrics, reset_metrics, ExecutionMetrics
from omniord.lazy import lazy_import


@pytest.fixture
def reset_state():
    """Reset all shared state before each test."""
    reset_metrics()
    reset_response_cache()
    yield
    reset_metrics()
    reset_response_cache()


def test_cache_hit_rate(reset_state):
    """Test that cache tracking reports correct hit rates."""
    metrics = get_metrics()
    
    # Simulate 3 cache hits and 2 misses
    for i in range(3):
        metric = ExecutionMetrics(
            operation="test",
            duration_ms=100.0,
            start_time=time.time(),
            cache_hit=True
        )
        metrics.record_execution(metric)
    
    for i in range(2):
        metric = ExecutionMetrics(
            operation="test",
            duration_ms=500.0,
            start_time=time.time(),
            cache_hit=False
        )
        metrics.record_execution(metric)
    
    # Verify hit rate calculation
    assert metrics.cache_hit_rate() == 60.0  # 3 / 5 * 100
    assert metrics.cache_hits == 3
    assert metrics.cache_misses == 2


def test_response_cache_lru_eviction(reset_state):
    """Test that cache evicts least-recently-used entries."""
    cache = get_response_cache()
    
    # Fill cache with small max size
    small_cache = type(cache)(max_size=3)
    
    small_cache.put("key1", "value1")
    small_cache.put("key2", "value2")
    small_cache.put("key3", "value3")
    
    # Access key1 to mark it as recently used
    small_cache.get("key1")
    
    # Add a new entry - should evict key2 (least recently used)
    small_cache.put("key4", "value4")
    
    assert small_cache.get("key1") == "value1"
    assert small_cache.get("key2") is None  # Evicted
    assert small_cache.get("key3") == "value3"
    assert small_cache.get("key4") == "value4"


def test_response_cache_with_pydantic_models(reset_state):
    """Test that cache works with Pydantic models."""
    from pydantic import BaseModel
    
    cache = get_response_cache()
    
    class Query(BaseModel):
        prompt: str
        model: str
    
    q1 = Query(prompt="Hello", model="llama")
    q2 = Query(prompt="Hello", model="llama")  # Same content
    
    cache.put("resp1", {"answer": "Hi"})
    
    # Different object, same content - should use same hash
    cached = cache.get("resp1")
    assert cached == {"answer": "Hi"}


def test_metrics_summary(reset_state):
    """Test that metrics summary provides complete information."""
    metrics = get_metrics()
    
    for i in range(5):
        metric = ExecutionMetrics(
            operation="llm_call",
            duration_ms=200.0 + i * 10,
            start_time=time.time(),
            input_tokens=10,
            output_tokens=50,
            cache_hit=(i % 2 == 0)
        )
        metrics.record_execution(metric)
    
    summary = metrics.summary()
    
    assert summary["executions"] == 5
    assert summary["total_input_tokens"] == 50
    assert summary["total_output_tokens"] == 250
    assert summary["total_tokens"] == 300
    assert 200 <= summary["avg_duration_ms"] <= 250
    assert summary["cache_hit_rate_pct"] == 60.0


def test_lazy_import():
    """Test that lazy imports work correctly."""
    # This should not raise even if modules aren't imported
    lazy_json = lazy_import("json")
    
    # Access should trigger import
    assert hasattr(lazy_json, "dumps")
    
    # Second access should be fast (already loaded)
    assert hasattr(lazy_json, "loads")


def test_cache_stats(reset_state):
    """Test cache statistics reporting."""
    cache = get_response_cache()
    
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.get("k1")  # Hit
    cache.get("k1")  # Hit
    
    stats = cache.stats()
    
    assert stats["entries"] == 2
    assert stats["total_hits"] == 2
    assert 0 < stats["utilization_pct"] < 1


def test_metrics_with_errors(reset_state):
    """Test that metrics can track errors."""
    metrics = get_metrics()
    
    metric = ExecutionMetrics(
        operation="failed_call",
        duration_ms=100.0,
        start_time=time.time(),
        error="Connection timeout"
    )
    metrics.record_execution(metric)
    
    assert len(metrics.metrics) == 1
    assert metrics.metrics[0].error == "Connection timeout"


@pytest.mark.asyncio
async def test_http_pool_reuse():
    """Test that HTTP pool reuses connections."""
    try:
        # Get client twice
        client1 = await get_http_client()
        client2 = await get_http_client()
        
        # Should be the same instance (singleton)
        assert client1 is client2
    finally:
        await close_http_pool()


def test_cache_ttl_expiration(reset_state):
    """Test that cached entries expire after TTL."""
    from omniord.cache import ResponseCache
    
    cache = ResponseCache(max_size=100, ttl_seconds=0.1)  # 100ms TTL
    
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Wait for TTL to expire
    time.sleep(0.15)
    
    # Should be expired now
    assert cache.get("key1") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
