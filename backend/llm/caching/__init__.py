"""Prompt Caching Module for Anthropic API.

This module provides utilities for leveraging Anthropic's prompt caching
feature to reduce costs and latency when using large, repeated context
like portfolio reports.

Prompt caching allows you to cache portions of your prompt (system prompts,
tools, large context) and reuse them across multiple API calls, reducing:
- Input token costs by up to 90%
- Latency for repeated calls

Usage:
    from backend.llm.caching import CachedClient, CacheMetrics

    client = CachedClient()
    response, metrics = client.create_message_with_cache(
        system_prompt=large_context,
        messages=messages,
        tools=tools
    )
    print(f"Cache hit: {metrics.cache_read_tokens} tokens")
"""

from backend.llm.caching.client import (
    CachedClient,
    CacheConfig,
    get_session_tracker,
    record_cache_metrics,
    get_session_stats,
)
from backend.llm.caching.metrics import (
    CacheMetrics,
    CacheStats,
    CacheStatsTracker,
)

__all__ = [
    "CachedClient",
    "CacheConfig",
    "CacheMetrics",
    "CacheStats",
    "CacheStatsTracker",
    "get_session_tracker",
    "record_cache_metrics",
    "get_session_stats",
]
