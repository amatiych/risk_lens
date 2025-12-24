"""Tests for the prompt caching module."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from backend.llm.caching import (
    CacheMetrics,
    CacheStats,
    CacheStatsTracker,
    CachedClient,
    CacheConfig,
)


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_cache_metrics_defaults(self):
        """Should create with default values."""
        metrics = CacheMetrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.cache_creation_tokens == 0
        assert metrics.cache_read_tokens == 0

    def test_cache_hit_detection(self):
        """Should detect cache hits."""
        hit_metrics = CacheMetrics(cache_read_tokens=100)
        miss_metrics = CacheMetrics(cache_creation_tokens=100)

        assert hit_metrics.cache_hit is True
        assert hit_metrics.cache_miss is False
        assert miss_metrics.cache_hit is False
        assert miss_metrics.cache_miss is True

    def test_cache_hit_ratio(self):
        """Should calculate cache hit ratio."""
        metrics = CacheMetrics(input_tokens=1000, cache_read_tokens=800)
        assert metrics.cache_hit_ratio == 0.8

    def test_cache_hit_ratio_zero_input(self):
        """Should handle zero input tokens."""
        metrics = CacheMetrics(input_tokens=0)
        assert metrics.cache_hit_ratio == 0.0

    def test_estimated_savings(self):
        """Should calculate estimated savings."""
        # 1000 input tokens, 800 cached (90% cheaper)
        # Regular: 1000 tokens at full price
        # Actual: 200 full price + 800 * 0.1 = 280
        # Savings: (1000 - 280) / 1000 = 72%
        metrics = CacheMetrics(input_tokens=1000, cache_read_tokens=800)
        assert metrics.estimated_savings_percent == pytest.approx(72.0, rel=0.01)

    def test_from_response(self):
        """Should create from API response."""
        mock_response = Mock()
        mock_response.usage = Mock(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=800,
            cache_read_input_tokens=0,
        )

        metrics = CacheMetrics.from_response(mock_response)

        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.cache_creation_tokens == 800
        assert metrics.cache_read_tokens == 0

    def test_from_response_no_usage(self):
        """Should handle response without usage."""
        mock_response = Mock(spec=[])  # No usage attribute

        metrics = CacheMetrics.from_response(mock_response)

        assert metrics.input_tokens == 0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = CacheMetrics(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=800,
        )
        d = metrics.to_dict()

        assert d["input_tokens"] == 1000
        assert d["cache_hit"] is True
        assert "timestamp" in d


class TestCacheStats:
    """Tests for CacheStats aggregation."""

    def test_cache_stats_defaults(self):
        """Should create with default values."""
        stats = CacheStats()
        assert stats.total_calls == 0
        assert stats.cache_hits == 0

    def test_cache_hit_rate(self):
        """Should calculate cache hit rate."""
        stats = CacheStats(total_calls=10, cache_hits=7)
        assert stats.cache_hit_rate == 70.0

    def test_cache_hit_rate_zero_calls(self):
        """Should handle zero calls."""
        stats = CacheStats()
        assert stats.cache_hit_rate == 0.0

    def test_token_cache_rate(self):
        """Should calculate token cache rate."""
        stats = CacheStats(
            total_input_tokens=10000,
            total_cached_tokens=8000,
        )
        assert stats.token_cache_rate == 80.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        stats = CacheStats(total_calls=5, cache_hits=3)
        d = stats.to_dict()

        assert d["total_calls"] == 5
        assert "cache_hit_rate" in d
        assert "estimated_savings" in d


class TestCacheStatsTracker:
    """Tests for CacheStatsTracker."""

    def test_record_metrics(self):
        """Should record metrics and update stats."""
        tracker = CacheStatsTracker()

        metrics = CacheMetrics(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=800,
        )
        tracker.record(metrics)

        stats = tracker.get_stats()
        assert stats.total_calls == 1
        assert stats.cache_hits == 1
        assert stats.total_input_tokens == 1000

    def test_record_multiple(self):
        """Should aggregate multiple recordings."""
        tracker = CacheStatsTracker()

        # First call - cache miss
        tracker.record(CacheMetrics(
            input_tokens=1000,
            cache_creation_tokens=800,
        ))

        # Second call - cache hit
        tracker.record(CacheMetrics(
            input_tokens=1000,
            cache_read_tokens=800,
        ))

        stats = tracker.get_stats()
        assert stats.total_calls == 2
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1

    def test_get_recent_metrics(self):
        """Should return recent metrics."""
        tracker = CacheStatsTracker()

        for i in range(5):
            tracker.record(CacheMetrics(input_tokens=i * 100))

        recent = tracker.get_recent_metrics(3)
        assert len(recent) == 3
        assert recent[-1].input_tokens == 400

    def test_get_last_metric(self):
        """Should return last metric."""
        tracker = CacheStatsTracker()
        tracker.record(CacheMetrics(input_tokens=100))
        tracker.record(CacheMetrics(input_tokens=200))

        last = tracker.get_last_metric()
        assert last.input_tokens == 200

    def test_get_last_metric_empty(self):
        """Should return None when empty."""
        tracker = CacheStatsTracker()
        assert tracker.get_last_metric() is None

    def test_reset(self):
        """Should reset all statistics."""
        tracker = CacheStatsTracker()
        tracker.record(CacheMetrics(input_tokens=1000))

        tracker.reset()

        stats = tracker.get_stats()
        assert stats.total_calls == 0
        assert tracker.get_last_metric() is None

    def test_to_json(self):
        """Should convert to JSON."""
        tracker = CacheStatsTracker()
        tracker.record(CacheMetrics(input_tokens=1000))

        json_str = tracker.to_json()
        assert "stats" in json_str
        assert "recent_metrics" in json_str


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.cache_system_prompt is True
        assert config.cache_tools is True
        assert config.min_cache_tokens == 1024

    def test_custom_config(self):
        """Should accept custom values."""
        config = CacheConfig(
            enabled=False,
            cache_tools=False,
            model="claude-opus-4-20250514",
        )

        assert config.enabled is False
        assert config.cache_tools is False
        assert config.model == "claude-opus-4-20250514"


class TestCachedClient:
    """Tests for CachedClient."""

    def test_build_cached_system(self):
        """Should format system prompt for caching."""
        client = CachedClient(api_key="test-key")

        result = client._build_cached_system("Test prompt")

        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Test prompt"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_build_cached_system_disabled(self):
        """Should not cache when disabled."""
        config = CacheConfig(enabled=False)
        client = CachedClient(api_key="test-key", config=config)

        result = client._build_cached_system("Test prompt")

        assert result == "Test prompt"

    def test_build_cached_tools(self):
        """Should add cache_control to last tool."""
        client = CachedClient(api_key="test-key")

        tools = [
            {"name": "tool1", "description": "First"},
            {"name": "tool2", "description": "Second"},
        ]

        result = client._build_cached_tools(tools)

        # First tool should not have cache_control
        assert "cache_control" not in result[0]
        # Last tool should have cache_control
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_build_cached_tools_empty(self):
        """Should handle empty tools list."""
        client = CachedClient(api_key="test-key")

        result = client._build_cached_tools([])
        assert result == []

    def test_build_cached_tools_disabled(self):
        """Should not cache tools when disabled."""
        config = CacheConfig(cache_tools=False)
        client = CachedClient(api_key="test-key", config=config)

        tools = [{"name": "tool1"}]
        result = client._build_cached_tools(tools)

        assert "cache_control" not in result[0]

    def test_get_stats(self):
        """Should return stats tracker."""
        client = CachedClient(api_key="test-key")

        tracker = client.get_stats()
        assert isinstance(tracker, CacheStatsTracker)

    def test_reset_stats(self):
        """Should reset statistics."""
        client = CachedClient(api_key="test-key")

        # Manually record some metrics
        client._tracker.record(CacheMetrics(input_tokens=100))
        assert client._tracker.get_stats().total_calls == 1

        client.reset_stats()
        assert client._tracker.get_stats().total_calls == 0

    @patch('backend.llm.caching.client.anthropic.Anthropic')
    def test_create_message_with_cache(self, mock_anthropic):
        """Should call API with cached format."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_response = Mock()
        mock_response.usage = Mock(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=800,
            cache_read_input_tokens=0,
        )
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response", type="text")]
        mock_client.messages.create.return_value = mock_response

        # Test
        client = CachedClient(api_key="test-key")
        response, metrics = client.create_message_with_cache(
            system_prompt="System prompt",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args

        # Check system prompt is formatted for caching
        system_arg = call_args.kwargs["system"]
        assert isinstance(system_arg, list)
        assert system_arg[0]["cache_control"] == {"type": "ephemeral"}

        # Check metrics
        assert metrics.input_tokens == 1000
        assert metrics.cache_creation_tokens == 800


class TestIntegration:
    """Integration tests for caching module."""

    def test_full_tracking_workflow(self):
        """Should track metrics across multiple calls."""
        tracker = CacheStatsTracker()

        # Simulate first call (cache miss)
        tracker.record(CacheMetrics(
            input_tokens=5000,
            output_tokens=500,
            cache_creation_tokens=4000,
        ))

        # Simulate subsequent calls (cache hits)
        for _ in range(3):
            tracker.record(CacheMetrics(
                input_tokens=5000,
                output_tokens=300,
                cache_read_tokens=4000,
            ))

        stats = tracker.get_stats()

        assert stats.total_calls == 4
        assert stats.cache_hits == 3
        assert stats.cache_misses == 1
        assert stats.cache_hit_rate == 75.0
        assert stats.total_cached_tokens == 12000  # 3 * 4000

    def test_savings_calculation_accuracy(self):
        """Should accurately calculate cost savings."""
        # Scenario: 10000 tokens, 8000 cached
        # Without cache: 10000 tokens at $3/1M = $0.03
        # With cache: 2000 at $3/1M + 8000 at $0.30/1M = $0.006 + $0.0024 = $0.0084
        # Savings: ($0.03 - $0.0084) / $0.03 = 72%

        metrics = CacheMetrics(
            input_tokens=10000,
            cache_read_tokens=8000,
        )

        assert metrics.estimated_savings_percent == pytest.approx(72.0, rel=0.01)
