"""Tests for the LLM tracing module."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from backend.llm.tracing import (
    Span,
    Trace,
    SpanKind,
    SpanStatus,
    Tracer,
    InMemoryStorage,
    TracedClient,
    traced_llm_call,
    traced_tool_call,
)
from backend.llm.tracing.core import TokenUsage
from backend.llm.tracing.storage import TraceQuery, TraceStats


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_token_usage_defaults(self):
        """Should create with default values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_total_tokens(self):
        """Should calculate total tokens."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_to_dict(self):
        """Should convert to dictionary."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["total_tokens"] == 150


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Should create span with required fields."""
        span = Span.create("test_span", "trace123", SpanKind.LLM)

        assert span.name == "test_span"
        assert span.trace_id == "trace123"
        assert span.kind == SpanKind.LLM
        assert span.status == SpanStatus.UNSET

    def test_span_duration(self):
        """Should calculate duration after end."""
        span = Span.create("test", "trace")
        time.sleep(0.01)  # 10ms
        span.end()

        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_span_attributes(self):
        """Should set attributes."""
        span = Span.create("test", "trace")
        span.set_attribute("key1", "value1")
        span.set_attributes({"key2": "value2", "key3": 123})

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"
        assert span.attributes["key3"] == 123

    def test_span_events(self):
        """Should add events."""
        span = Span.create("test", "trace")
        span.add_event("event1", {"data": "value"})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "event1"

    def test_span_tokens(self):
        """Should set token usage."""
        span = Span.create("test", "trace")
        span.set_tokens(input_tokens=100, output_tokens=50)

        assert span.tokens is not None
        assert span.tokens.input_tokens == 100
        assert span.tokens.output_tokens == 50

    def test_span_tokens_from_usage(self):
        """Should extract tokens from API response."""
        span = Span.create("test", "trace")

        mock_usage = Mock(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=800,
            cache_creation_input_tokens=0,
        )
        span.set_tokens_from_usage(mock_usage)

        assert span.tokens.input_tokens == 1000
        assert span.tokens.cache_read_tokens == 800

    def test_span_error(self):
        """Should record exceptions."""
        span = Span.create("test", "trace")

        try:
            raise ValueError("test error")
        except Exception as e:
            span.record_exception(e)

        assert span.status == SpanStatus.ERROR
        assert span.error is not None
        assert span.error["type"] == "ValueError"
        assert "test error" in span.error["message"]

    def test_span_context_manager(self):
        """Should work as context manager."""
        with Span.create("test", "trace") as span:
            span.set_attribute("key", "value")

        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_span_context_manager_error(self):
        """Should capture errors in context manager."""
        with pytest.raises(ValueError):
            with Span.create("test", "trace") as span:
                raise ValueError("test")

        assert span.status == SpanStatus.ERROR

    def test_span_to_dict(self):
        """Should convert to dictionary."""
        span = Span.create("test", "trace", SpanKind.LLM)
        span.set_tokens(input_tokens=100, output_tokens=50)
        span.end()

        d = span.to_dict()

        assert d["name"] == "test"
        assert d["kind"] == "llm"
        assert d["tokens"]["input_tokens"] == 100


class TestTrace:
    """Tests for Trace class."""

    def test_trace_creation(self):
        """Should create trace."""
        trace = Trace.create("test_trace")

        assert trace.name == "test_trace"
        assert trace.trace_id is not None
        assert trace.status == SpanStatus.UNSET

    def test_trace_span(self):
        """Should create spans."""
        trace = Trace.create("test")

        with trace.span("span1", SpanKind.LLM) as span:
            span.set_tokens(input_tokens=100, output_tokens=50)

        assert len(trace.spans) == 1
        assert trace.spans[0].name == "span1"

    def test_trace_nested_spans(self):
        """Should support nested spans."""
        trace = Trace.create("test")

        with trace.span("outer") as outer:
            with trace.span("inner") as inner:
                pass

        assert len(trace.spans) == 2
        assert trace.spans[1].parent_span_id == trace.spans[0].span_id

    def test_trace_total_tokens(self):
        """Should aggregate tokens."""
        trace = Trace.create("test")

        with trace.span("span1", SpanKind.LLM) as span:
            span.set_tokens(input_tokens=100, output_tokens=50)

        with trace.span("span2", SpanKind.LLM) as span:
            span.set_tokens(input_tokens=200, output_tokens=100)

        tokens = trace.total_tokens
        assert tokens.input_tokens == 300
        assert tokens.output_tokens == 150

    def test_trace_has_error(self):
        """Should detect errors in spans."""
        trace = Trace.create("test")

        with trace.span("ok_span"):
            pass

        with pytest.raises(ValueError):
            with trace.span("error_span"):
                raise ValueError("test")

        assert trace.has_error is True

    def test_trace_llm_spans(self):
        """Should filter LLM spans."""
        trace = Trace.create("test")

        trace.span("llm1", SpanKind.LLM).end()
        trace.span("tool1", SpanKind.TOOL).end()
        trace.span("llm2", SpanKind.LLM).end()

        assert len(trace.llm_spans) == 2
        assert len(trace.tool_spans) == 1

    def test_trace_context_manager(self):
        """Should work as context manager."""
        with Trace.create("test") as trace:
            trace.set_attribute("key", "value")

        assert trace.status == SpanStatus.OK
        assert trace.end_time is not None

    def test_trace_to_dict(self):
        """Should convert to dictionary."""
        with Trace.create("test") as trace:
            with trace.span("span1", SpanKind.LLM) as span:
                span.set_tokens(input_tokens=100, output_tokens=50)

        d = trace.to_dict()

        assert d["name"] == "test"
        assert len(d["spans"]) == 1
        assert d["total_tokens"]["input_tokens"] == 100

    def test_trace_summary(self):
        """Should generate readable summary."""
        with Trace.create("test_trace") as trace:
            with trace.span("span1", SpanKind.LLM) as span:
                span.set_tokens(input_tokens=100, output_tokens=50)

        summary = trace.summary()

        assert "test_trace" in summary
        assert "100" in summary


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryStorage(max_traces=100)

    def test_save_and_get(self, storage):
        """Should save and retrieve traces."""
        trace = Trace.create("test")
        trace.end()

        storage.save(trace)
        retrieved = storage.get(trace.trace_id)

        assert retrieved is not None
        assert retrieved.trace_id == trace.trace_id

    def test_get_recent(self, storage):
        """Should return recent traces."""
        for i in range(5):
            trace = Trace.create(f"trace{i}")
            trace.end()
            storage.save(trace)

        recent = storage.get_recent(3)

        assert len(recent) == 3
        assert recent[0].name == "trace4"  # Most recent first

    def test_query_by_name(self, storage):
        """Should filter by name."""
        for name in ["chat_request", "analysis", "chat_response"]:
            trace = Trace.create(name)
            trace.end()
            storage.save(trace)

        results = storage.query(TraceQuery(name="chat"))

        assert len(results) == 2

    def test_query_by_error(self, storage):
        """Should filter by error status."""
        ok_trace = Trace.create("ok")
        ok_trace.end()
        storage.save(ok_trace)

        # Create error trace with a failed span
        error_trace = Trace.create("error")
        with error_trace.span("failed_span") as span:
            span.status = SpanStatus.ERROR
        error_trace.end()
        storage.save(error_trace)

        results = storage.query(TraceQuery(has_error=True))

        assert len(results) == 1
        assert results[0].name == "error"

    def test_query_by_duration(self, storage):
        """Should filter by duration."""
        fast = Trace.create("fast")
        fast.end()
        storage.save(fast)

        slow = Trace.create("slow")
        time.sleep(0.05)
        slow.end()
        storage.save(slow)

        results = storage.query(TraceQuery(min_duration_ms=40))

        assert len(results) == 1
        assert results[0].name == "slow"

    def test_max_capacity(self):
        """Should evict old traces at capacity."""
        storage = InMemoryStorage(max_traces=5)

        for i in range(10):
            trace = Trace.create(f"trace{i}")
            trace.end()
            storage.save(trace)

        # Should only have last 5
        recent = storage.get_recent(10)
        assert len(recent) == 5
        assert recent[0].name == "trace9"

    def test_get_stats(self, storage):
        """Should calculate statistics."""
        for i in range(3):
            with Trace.create(f"trace{i}") as trace:
                with trace.span("llm", SpanKind.LLM) as span:
                    span.set_tokens(input_tokens=100, output_tokens=50)
            storage.save(trace)

        stats = storage.get_stats()

        assert stats.total_traces == 3
        assert stats.total_llm_calls == 3
        assert stats.total_input_tokens == 300

    def test_clear(self, storage):
        """Should clear all traces."""
        storage.save(Trace.create("test"))
        storage.clear()

        assert len(storage.get_recent(10)) == 0


class TestTracer:
    """Tests for Tracer class."""

    @pytest.fixture
    def tracer(self):
        """Create tracer with fresh storage."""
        return Tracer(storage=InMemoryStorage())

    def test_trace_context(self, tracer):
        """Should manage trace context."""
        with tracer.trace("test_trace") as trace:
            assert tracer.get_active_trace() == trace

        assert tracer.get_active_trace() is None

    def test_trace_saved(self, tracer):
        """Should save trace to storage."""
        with tracer.trace("test"):
            pass

        traces = tracer.get_recent_traces(10)
        assert len(traces) == 1

    def test_span_in_trace(self, tracer):
        """Should create spans in active trace."""
        with tracer.trace("test") as trace:
            with tracer.span("span1", SpanKind.LLM):
                pass

        assert len(trace.spans) == 1

    def test_disabled_tracer(self):
        """Should no-op when disabled."""
        tracer = Tracer(enabled=False, storage=InMemoryStorage())

        with tracer.trace("test"):
            with tracer.span("span"):
                pass

        # Should not save anything
        assert len(tracer.get_recent_traces(10)) == 0

    def test_get_stats(self, tracer):
        """Should return stats from storage."""
        with tracer.trace("test"):
            pass

        stats = tracer.get_stats()

        assert stats["total_traces"] == 1


class TestInstrumentation:
    """Tests for instrumentation decorators."""

    def test_traced_llm_call_decorator(self):
        """Should trace decorated function."""
        tracer = Tracer(storage=InMemoryStorage())

        @traced_llm_call(tracer=tracer)
        def mock_llm_call():
            response = Mock()
            response.usage = Mock(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
            response.model = "test-model"
            return response

        with tracer.trace("test"):
            result = mock_llm_call()

        traces = tracer.get_recent_traces(1)
        assert len(traces) == 1
        assert len(traces[0].spans) == 1
        assert traces[0].spans[0].tokens.input_tokens == 100

    def test_traced_tool_call_decorator(self):
        """Should trace tool execution."""
        tracer = Tracer(storage=InMemoryStorage())

        @traced_tool_call(tracer=tracer)
        def mock_tool(arg1, arg2):
            return {"result": "success"}

        with tracer.trace("test"):
            result = mock_tool("a", "b")

        traces = tracer.get_recent_traces(1)
        assert len(traces[0].spans) == 1
        assert traces[0].spans[0].kind == SpanKind.TOOL

    def test_traced_decorator_error(self):
        """Should capture errors in decorated functions."""
        tracer = Tracer(storage=InMemoryStorage())

        @traced_llm_call(tracer=tracer)
        def failing_call():
            raise RuntimeError("API error")

        with tracer.trace("test"):
            with pytest.raises(RuntimeError):
                failing_call()

        traces = tracer.get_recent_traces(1)
        assert traces[0].spans[0].status == SpanStatus.ERROR


class TestTracedClient:
    """Tests for TracedClient wrapper."""

    def test_traced_client_create(self):
        """Should trace messages.create calls."""
        tracer = Tracer(storage=InMemoryStorage())

        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.usage = Mock(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_client.messages.create.return_value = mock_response

        traced = TracedClient(mock_client, tracer=tracer)

        with tracer.trace("test"):
            response = traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello"}]
            )

        traces = tracer.get_recent_traces(1)
        span = traces[0].spans[0]

        assert span.kind == SpanKind.LLM
        assert span.tokens.input_tokens == 1000
        assert span.attributes["model"] == "claude-sonnet-4-20250514"


class TestIntegration:
    """Integration tests for complete tracing workflow."""

    def test_full_chat_trace(self):
        """Should trace a complete chat interaction."""
        tracer = Tracer(storage=InMemoryStorage())

        with tracer.trace("chat_request") as trace:
            trace.set_attribute("user_id", "user123")

            # Input validation
            with tracer.span("input_validation", SpanKind.GUARDRAIL) as span:
                span.set_attribute("passed", True)

            # LLM call
            with tracer.span("llm_call", SpanKind.LLM) as span:
                span.set_tokens(input_tokens=1000, output_tokens=500)
                span.set_attribute("model", "claude-sonnet-4-20250514")

            # Tool call
            with tracer.span("tool:get_price", SpanKind.TOOL) as span:
                span.set_attribute("ticker", "AAPL")

            # Output validation
            with tracer.span("output_validation", SpanKind.GUARDRAIL) as span:
                span.set_attribute("disclaimer_added", True)

        # Verify trace
        traces = tracer.get_recent_traces(1)
        assert len(traces) == 1

        trace = traces[0]
        assert trace.name == "chat_request"
        assert len(trace.spans) == 4
        assert trace.total_tokens.input_tokens == 1000
        assert trace.attributes["user_id"] == "user123"

        # Verify stats
        stats = tracer.get_stats()
        assert stats["total_traces"] == 1
        assert stats["total_llm_calls"] == 1
        assert stats["total_tool_calls"] == 1
