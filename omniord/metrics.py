"""Performance metrics and observability instrumentation for Omniord.

Tracks execution timing, token usage, cache hit rates, and resource utilization
across the entire orchestration pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class ExecutionMetrics(BaseModel):
    """Track timing and resource usage for a single execution."""

    operation: str
    duration_ms: float
    start_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True


@dataclass
class MetricsCollector:
    """Centralized metrics collection for performance monitoring."""

    metrics: list[ExecutionMetrics] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def record_execution(self, metric: ExecutionMetrics) -> None:
        """Record a single execution metric."""
        self.metrics.append(metric)
        self.total_input_tokens += metric.input_tokens
        self.total_output_tokens += metric.output_tokens

        if metric.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def cache_hit_rate(self) -> float:
        """Return cache hit rate as a percentage (0-100)."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    def avg_duration_ms(self) -> float:
        """Return average execution duration in milliseconds."""
        if not self.metrics:
            return 0.0
        return sum(m.duration_ms for m in self.metrics) / len(self.metrics)

    def total_tokens(self) -> int:
        """Return total tokens used across all executions."""
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> dict[str, Any]:
        """Return a summary of all collected metrics."""
        return {
            "executions": len(self.metrics),
            "cache_hit_rate_pct": self.cache_hit_rate(),
            "avg_duration_ms": self.avg_duration_ms(),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens(),
        }


# Process-wide metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the process-wide metrics collector."""
    return _metrics


def reset_metrics() -> None:
    """Reset metrics (useful for tests)."""
    global _metrics
    _metrics = MetricsCollector()
