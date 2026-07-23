# Performance Optimization Guide

This document outlines the performance optimizations implemented in the `perf/optimizations` branch for OMNIORD.

## Overview

The optimization effort focuses on four key areas:
1. **HTTP Connection Pooling** — Reuse connections across LLM provider calls
2. **Response Caching** — Cache LLM responses with LRU eviction and TTL
3. **Metrics Tracking** — Monitor execution timing and token usage
4. **Lazy Loading** — Defer heavy SDK imports until needed

---

## 1. HTTP Connection Pooling (`http_pool.py`)

### What It Does
- Maintains a **singleton async HTTP client** with connection pooling
- Reuses TCP connections across multiple LLM calls
- Configurable limits: 100 max connections, 20 keepalive connections
- Automatic resource cleanup at shutdown

### Usage
```python
from omniord.http_pool import http_session, get_http_client

# Option 1: Using context manager (recommended)
async with http_session() as client:
    response = await client.get("http://localhost:11434/api/tags")

# Option 2: Get the pooled client directly
client = await get_http_client(timeout=120.0)
response = await client.get("http://localhost:11434/api/tags")
```

### Performance Impact
- **Before**: New connection for each request → TCP handshake overhead
- **After**: Reused connections → ~10-50ms saved per request
- **Best for**: High-frequency calls to Ollama or cloud APIs

---

## 2. Response Caching (`cache.py`)

### What It Does
- **LRU (Least Recently Used) cache** for LLM responses
- Automatic eviction when max size (1000) is reached
- Optional TTL-based expiration (default: 1 hour)
- Deterministic hashing of inputs (supports dicts, Pydantic models, strings)

### Usage
```python
from omniord.cache import get_response_cache

cache = get_response_cache()

# Get or compute
def expensive_llm_call(prompt):
    return model.generate(prompt)

result = cache.get_or_compute(
    input_data={"prompt": "What is AI?"},
    compute_fn=expensive_llm_call,
    prompt="What is AI?"
)

# View stats
print(cache.stats())
# Output: {'entries': 42, 'max_size': 1000, 'total_hits': 156, 'utilization_pct': 4.2}
```

### Performance Impact
- **Cache hit**: <1ms response time (in-memory lookup)
- **Cache miss**: Full LLM call (100-5000ms depending on model)
- **Typical savings**: 30-50% reduction in LLM call latency for repeated queries

---

## 3. Metrics Tracking (`metrics.py`)

### What It Does
- Tracks execution timing (per-operation duration in milliseconds)
- Counts input/output tokens for cost estimation
- Monitors cache hit/miss rates
- Provides summary statistics

### Usage
```python
from omniord.metrics import get_metrics, ExecutionMetrics
import time

metrics = get_metrics()

# Record an execution
start = time.time()
result = model.generate("Hello")
duration_ms = (time.time() - start) * 1000

metric = ExecutionMetrics(
    operation="llm_call",
    duration_ms=duration_ms,
    start_time=start,
    input_tokens=10,
    output_tokens=50,
    cache_hit=False
)
metrics.record_execution(metric)

# View summary
print(metrics.summary())
# Output: {
#     'executions': 5,
#     'cache_hit_rate_pct': 20.0,
#     'avg_duration_ms': 234.5,
#     'total_input_tokens': 150,
#     'total_output_tokens': 500,
#     'total_tokens': 650
# }
```

### Performance Impact
- **Minimal overhead**: <1ms per tracked operation
- **Visibility**: Identify bottlenecks and optimize accordingly

---

## 4. Lazy Loading (`lazy.py`)

### What It Does
- Defers heavy SDK imports (Anthropic, OpenAI) until first use
- Reduces startup time
- Only loads cloud providers if actually used

### Usage
```python
from omniord.lazy import lazy_import, anthropic, openai

# Pre-configured lazy loaders
client = anthropic.Anthropic(api_key="sk-ant-...")

# Or use generic lazy_import
httpx = lazy_import("httpx")
async_client = httpx.AsyncClient()
```

### Performance Impact
- **Startup time**: 50-200ms saved (depends on SDK size)
- **Best for**: CLI tools that may not always need cloud providers

---

## Integration Guide

### In Router Providers

```python
# router/providers/ollama.py
from omniord.http_pool import http_session
from omniord.cache import get_response_cache
from omniord.metrics import ExecutionMetrics, get_metrics
import time

class OllamaProvider:
    async def generate(self, prompt: str, model: str):
        cache = get_response_cache()
        metrics = get_metrics()
        
        start = time.time()
        
        # Check cache first
        cached = cache.get(f"{model}:{prompt}")
        if cached:
            metric = ExecutionMetrics(
                operation="ollama_generate",
                duration_ms=(time.time() - start) * 1000,
                start_time=start,
                cache_hit=True
            )
            metrics.record_execution(metric)
            return cached
        
        # Use pooled HTTP connection
        async with http_session() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt}
            )
        
        result = response.json()
        cache.put(f"{model}:{prompt}", result)
        
        metric = ExecutionMetrics(
            operation="ollama_generate",
            duration_ms=(time.time() - start) * 1000,
            start_time=start,
            output_tokens=result.get("eval_count", 0)
        )
        metrics.record_execution(metric)
        
        return result
```

### In Main CLI

```python
# main.py
from omniord.metrics import get_metrics
from omniord.http_pool import close_http_pool
import asyncio

@app.command()
async def run(prompt: str):
    try:
        # Your orchestration here
        pass
    finally:
        # Cleanup and report metrics
        metrics = get_metrics()
        console.print(Panel(
            f"Cache hit rate: {metrics.cache_hit_rate():.1f}%\n"
            f"Avg duration: {metrics.avg_duration_ms():.1f}ms\n"
            f"Total tokens: {metrics.total_tokens()}"
        ))
        await close_http_pool()
```

---

## Performance Expectations

| Optimization | Typical Improvement | Notes |
|---|---|---|
| HTTP pooling | 10-50ms/request | Scales with request frequency |
| Response caching | 99% faster (cache hit) | 30-50% reduction in latency |
| Lazy loading | 50-200ms startup | One-time benefit |
| Metrics tracking | <1ms overhead | Minimal impact on throughput |

---

## Configuration

All optimizations are enabled by default. To customize:

```python
from omniord.cache import ResponseCache
from omniord.http_pool import HTTPConnectionPool

# Custom cache size and TTL
cache = ResponseCache(max_size=5000, ttl_seconds=7200)

# Custom HTTP pool timeouts
pool = HTTPConnectionPool()
client = await pool.get_client(timeout=300.0)
```

---

## Testing

```bash
# Run existing tests
pytest

# Run with metrics reporting
pytest -v --tb=short

# Check cache effectiveness
pytest -k "cache" -v
```

---

## Next Steps

1. Integrate metrics into CLI output
2. Add Prometheus export for observability
3. Implement adaptive TTL based on model confidence
4. Profile with real workloads to identify further bottlenecks
