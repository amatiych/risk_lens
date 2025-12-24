"""Cached Anthropic Client with prompt caching support.

This module provides a wrapper around the Anthropic client that
automatically applies prompt caching to reduce costs and latency.

Anthropic's prompt caching works by:
1. Marking portions of the prompt with cache_control: {"type": "ephemeral"}
2. The first call writes these portions to cache (cache_creation_input_tokens)
3. Subsequent calls within 5 minutes read from cache (cache_read_input_tokens)
4. Cached tokens cost 90% less than regular input tokens

Best practices:
- Cache large, static content (system prompts, tools, documents)
- Place cached content at the beginning of the prompt
- Minimum cacheable size is 1024 tokens for Claude 3.5 Sonnet
- Cache has 5-minute TTL, refreshed on each use
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Generator
import anthropic

from backend.llm.caching.metrics import CacheMetrics, CacheStatsTracker


@dataclass
class CacheConfig:
    """Configuration for prompt caching behavior.

    Attributes:
        enabled: Whether caching is enabled.
        cache_system_prompt: Whether to cache the system prompt.
        cache_tools: Whether to cache tool definitions.
        min_cache_tokens: Minimum tokens to enable caching (default 1024).
        model: Model to use.
    """
    enabled: bool = True
    cache_system_prompt: bool = True
    cache_tools: bool = True
    min_cache_tokens: int = 1024
    model: str = "claude-sonnet-4-20250514"


class CachedClient:
    """Anthropic client wrapper with automatic prompt caching.

    This client automatically applies cache_control to system prompts
    and tools, and tracks cache metrics for monitoring.

    Example:
        client = CachedClient()

        # Simple usage
        response, metrics = client.create_message_with_cache(
            system_prompt="You are a helpful assistant...",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Check cache performance
        print(f"Cache hit: {metrics.cache_hit}")
        print(f"Savings: {metrics.estimated_savings_percent:.1f}%")

        # Get session stats
        stats = client.get_stats()
        print(f"Session cache hit rate: {stats.cache_hit_rate:.1f}%")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[CacheConfig] = None
    ):
        """Initialize the cached client.

        Args:
            api_key: Anthropic API key. Uses CLAUDE_API_KEY env var if not provided.
            config: Cache configuration. Uses defaults if not provided.
        """
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.config = config or CacheConfig()
        self._client: Optional[anthropic.Anthropic] = None
        self._tracker = CacheStatsTracker()

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("No API key provided. Set CLAUDE_API_KEY environment variable.")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _build_cached_system(self, system_prompt: str) -> List[Dict[str, Any]]:
        """Build system prompt with cache control.

        Args:
            system_prompt: The system prompt text.

        Returns:
            System prompt formatted for caching.
        """
        if not self.config.enabled or not self.config.cache_system_prompt:
            return system_prompt

        # For caching, system must be a list of content blocks
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]

    def _build_cached_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build tools with cache control on the last tool.

        Args:
            tools: List of tool definitions.

        Returns:
            Tools with cache control applied.
        """
        if not tools or not self.config.enabled or not self.config.cache_tools:
            return tools

        # Apply cache_control to the last tool (Anthropic recommendation)
        cached_tools = []
        for i, tool in enumerate(tools):
            tool_copy = dict(tool)
            if i == len(tools) - 1:
                tool_copy["cache_control"] = {"type": "ephemeral"}
            cached_tools.append(tool_copy)

        return cached_tools

    def create_message_with_cache(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2000,
        **kwargs
    ) -> Tuple[Any, CacheMetrics]:
        """Create a message with prompt caching enabled.

        Args:
            system_prompt: System prompt to cache.
            messages: Conversation messages.
            tools: Optional tool definitions to cache.
            max_tokens: Maximum response tokens.
            **kwargs: Additional arguments passed to messages.create().

        Returns:
            Tuple of (response, CacheMetrics).
        """
        # Build cached system prompt
        cached_system = self._build_cached_system(system_prompt)

        # Build request parameters
        request_params = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": cached_system,
            "messages": messages,
            **kwargs
        }

        # Add cached tools if provided
        if tools:
            request_params["tools"] = self._build_cached_tools(tools)

        # Make the API call
        response = self.client.messages.create(**request_params)

        # Extract metrics
        metrics = CacheMetrics.from_response(response, self.config.model)
        self._tracker.record(metrics)

        return response, metrics

    def stream_message_with_cache(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2000,
        **kwargs
    ) -> Generator[Tuple[str, Optional[CacheMetrics]], None, None]:
        """Stream a message with prompt caching enabled.

        Args:
            system_prompt: System prompt to cache.
            messages: Conversation messages.
            tools: Optional tool definitions to cache.
            max_tokens: Maximum response tokens.
            **kwargs: Additional arguments passed to messages.stream().

        Yields:
            Tuples of (text_chunk, None) during streaming,
            then ("", CacheMetrics) at the end with final metrics.
        """
        # Build cached system prompt
        cached_system = self._build_cached_system(system_prompt)

        # Build request parameters
        request_params = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": cached_system,
            "messages": messages,
            **kwargs
        }

        # Add cached tools if provided
        if tools:
            request_params["tools"] = self._build_cached_tools(tools)

        # Stream the response
        with self.client.messages.stream(**request_params) as stream:
            for text in stream.text_stream:
                yield text, None

            # Get final message for metrics
            final_message = stream.get_final_message()
            metrics = CacheMetrics.from_response(final_message, self.config.model)
            self._tracker.record(metrics)

            yield "", metrics

    def get_stats(self) -> "CacheStatsTracker":
        """Get the cache statistics tracker."""
        return self._tracker

    def get_last_metrics(self) -> Optional[CacheMetrics]:
        """Get metrics from the last API call."""
        return self._tracker.get_last_metric()

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._tracker.reset()


# Singleton instance for session-wide tracking
_default_tracker = CacheStatsTracker()


def get_session_tracker() -> CacheStatsTracker:
    """Get the session-wide cache tracker.

    Returns:
        The global CacheStatsTracker instance.
    """
    return _default_tracker


def record_cache_metrics(metrics: CacheMetrics) -> None:
    """Record metrics to the session tracker.

    Args:
        metrics: CacheMetrics to record.
    """
    _default_tracker.record(metrics)


def get_session_stats() -> Dict[str, Any]:
    """Get session cache statistics.

    Returns:
        Dictionary with cache statistics.
    """
    return _default_tracker.get_stats().to_dict()
