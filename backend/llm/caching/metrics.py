"""Cache metrics tracking for prompt caching.

This module provides classes for tracking and analyzing cache performance
metrics from Anthropic API responses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class CacheMetrics:
    """Metrics from a single cached API call.

    Anthropic's API returns cache statistics in the usage field:
    - cache_creation_input_tokens: Tokens written to cache (first call)
    - cache_read_input_tokens: Tokens read from cache (subsequent calls)

    Attributes:
        input_tokens: Total input tokens for this call.
        output_tokens: Total output tokens for this call.
        cache_creation_tokens: Tokens written to cache (cache miss).
        cache_read_tokens: Tokens read from cache (cache hit).
        timestamp: When this call was made.
        model: Model used for the call.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    model: str = "claude-sonnet-4-20250514"

    @property
    def cache_hit(self) -> bool:
        """Whether this call had a cache hit."""
        return self.cache_read_tokens > 0

    @property
    def cache_miss(self) -> bool:
        """Whether this call created new cache entries."""
        return self.cache_creation_tokens > 0

    @property
    def cache_hit_ratio(self) -> float:
        """Ratio of cached tokens to total input tokens."""
        if self.input_tokens == 0:
            return 0.0
        return self.cache_read_tokens / self.input_tokens

    @property
    def estimated_savings_percent(self) -> float:
        """Estimated cost savings from caching.

        Cached tokens cost 90% less than regular input tokens.
        """
        if self.input_tokens == 0:
            return 0.0
        # Cached tokens cost 10% of regular price (90% savings)
        regular_cost = self.input_tokens
        actual_cost = (self.input_tokens - self.cache_read_tokens) + (self.cache_read_tokens * 0.1)
        if regular_cost == 0:
            return 0.0
        return ((regular_cost - actual_cost) / regular_cost) * 100

    @classmethod
    def from_response(cls, response: Any, model: str = "claude-sonnet-4-20250514") -> "CacheMetrics":
        """Create CacheMetrics from an Anthropic API response.

        Args:
            response: The response object from anthropic.messages.create()
            model: Model name used.

        Returns:
            CacheMetrics populated from the response usage data.
        """
        usage = getattr(response, 'usage', None)
        if usage is None:
            return cls(model=model)

        return cls(
            input_tokens=getattr(usage, 'input_tokens', 0),
            output_tokens=getattr(usage, 'output_tokens', 0),
            cache_creation_tokens=getattr(usage, 'cache_creation_input_tokens', 0),
            cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0),
            model=model,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_hit": self.cache_hit,
            "cache_hit_ratio": self.cache_hit_ratio,
            "estimated_savings_percent": self.estimated_savings_percent,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
        }


@dataclass
class CacheStats:
    """Aggregated cache statistics across multiple calls.

    Attributes:
        total_calls: Total number of API calls.
        cache_hits: Number of calls with cache hits.
        cache_misses: Number of calls that created cache entries.
        total_input_tokens: Total input tokens across all calls.
        total_output_tokens: Total output tokens across all calls.
        total_cached_tokens: Total tokens read from cache.
        total_cache_created_tokens: Total tokens written to cache.
    """
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cache_created_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Percentage of calls with cache hits."""
        if self.total_calls == 0:
            return 0.0
        return (self.cache_hits / self.total_calls) * 100

    @property
    def token_cache_rate(self) -> float:
        """Percentage of input tokens served from cache."""
        if self.total_input_tokens == 0:
            return 0.0
        return (self.total_cached_tokens / self.total_input_tokens) * 100

    @property
    def estimated_total_savings_percent(self) -> float:
        """Estimated total cost savings from caching."""
        if self.total_input_tokens == 0:
            return 0.0
        regular_cost = self.total_input_tokens
        actual_cost = (self.total_input_tokens - self.total_cached_tokens) + (self.total_cached_tokens * 0.1)
        return ((regular_cost - actual_cost) / regular_cost) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{self.cache_hit_rate:.1f}%",
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "token_cache_rate": f"{self.token_cache_rate:.1f}%",
            "estimated_savings": f"{self.estimated_total_savings_percent:.1f}%",
        }


class CacheStatsTracker:
    """Tracks cache statistics across a session.

    This class maintains running statistics for cache performance
    and can be used to display cache status in the UI.
    """

    def __init__(self):
        """Initialize the tracker."""
        self._metrics: List[CacheMetrics] = []
        self._stats = CacheStats()

    def record(self, metrics: CacheMetrics) -> None:
        """Record metrics from an API call.

        Args:
            metrics: CacheMetrics from a single call.
        """
        self._metrics.append(metrics)

        self._stats.total_calls += 1
        self._stats.total_input_tokens += metrics.input_tokens
        self._stats.total_output_tokens += metrics.output_tokens
        self._stats.total_cached_tokens += metrics.cache_read_tokens
        self._stats.total_cache_created_tokens += metrics.cache_creation_tokens

        if metrics.cache_hit:
            self._stats.cache_hits += 1
        if metrics.cache_miss:
            self._stats.cache_misses += 1

    def get_stats(self) -> CacheStats:
        """Get aggregated statistics."""
        return self._stats

    def get_recent_metrics(self, n: int = 10) -> List[CacheMetrics]:
        """Get the most recent N metrics.

        Args:
            n: Number of recent metrics to return.

        Returns:
            List of recent CacheMetrics.
        """
        return self._metrics[-n:]

    def get_last_metric(self) -> Optional[CacheMetrics]:
        """Get the most recent metric."""
        if not self._metrics:
            return None
        return self._metrics[-1]

    def reset(self) -> None:
        """Reset all statistics."""
        self._metrics.clear()
        self._stats = CacheStats()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stats": self._stats.to_dict(),
            "recent_metrics": [m.to_dict() for m in self.get_recent_metrics(5)],
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
