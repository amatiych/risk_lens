"""Tracer for creating and managing traces.

This module provides the main Tracer class for instrumenting LLM
operations with traces and spans.
"""

from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime
import threading

from backend.llm.tracing.core import Trace, Span, SpanKind, SpanStatus
from backend.llm.tracing.storage import TraceStorage, InMemoryStorage, get_trace_storage


class Tracer:
    """Main tracer for creating and managing traces.

    The Tracer creates traces and spans, and automatically saves
    completed traces to storage.

    Example:
        tracer = Tracer()

        with tracer.trace("chat_request") as trace:
            trace.set_attribute("user_id", "user123")

            with trace.span("input_validation", SpanKind.GUARDRAIL) as span:
                # validate input
                span.set_attribute("input_length", len(user_input))

            with trace.span("llm_call", SpanKind.LLM) as span:
                response = client.messages.create(...)
                span.set_tokens_from_usage(response.usage)
                span.set_attribute("model", "claude-sonnet-4-20250514")

    Attributes:
        name: Name of this tracer instance.
        storage: Storage backend for traces.
        enabled: Whether tracing is enabled.
    """

    def __init__(
        self,
        name: str = "default",
        storage: Optional[TraceStorage] = None,
        enabled: bool = True,
    ):
        """Initialize the tracer.

        Args:
            name: Tracer instance name.
            storage: Storage backend. Uses global if None.
            enabled: Whether to record traces.
        """
        self.name = name
        self.storage = storage or get_trace_storage()
        self.enabled = enabled
        self._active_trace: Optional[Trace] = None
        self._lock = threading.Lock()

    @contextmanager
    def trace(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Trace, None, None]:
        """Create and manage a trace context.

        Args:
            name: Trace name.
            attributes: Optional initial attributes.

        Yields:
            The active Trace.
        """
        if not self.enabled:
            # Return a no-op trace
            yield Trace.create(name)
            return

        trace = Trace.create(name)
        if attributes:
            trace.attributes.update(attributes)

        with self._lock:
            previous_trace = self._active_trace
            self._active_trace = trace

        try:
            yield trace
        finally:
            trace.end()
            self.storage.save(trace)

            with self._lock:
                self._active_trace = previous_trace

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Create a span in the current trace.

        If no trace is active, creates a standalone span.

        Args:
            name: Span name.
            kind: Span type.
            attributes: Optional initial attributes.

        Yields:
            The active Span.
        """
        if not self.enabled:
            yield Span.create(name, "noop", kind)
            return

        trace = self._active_trace
        if trace:
            span = trace.span(name, kind)
        else:
            # Standalone span
            span = Span.create(name, "standalone", kind)

        if attributes:
            span.attributes.update(attributes)

        try:
            yield span
        finally:
            span.end()

    def get_active_trace(self) -> Optional[Trace]:
        """Get the currently active trace.

        Returns:
            The active Trace or None.
        """
        with self._lock:
            return self._active_trace

    def get_recent_traces(self, limit: int = 10) -> List[Trace]:
        """Get recent traces from storage.

        Args:
            limit: Maximum traces to return.

        Returns:
            List of recent traces.
        """
        return self.storage.get_recent(limit)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a specific trace by ID.

        Args:
            trace_id: The trace ID.

        Returns:
            The Trace or None.
        """
        return self.storage.get(trace_id)

    def get_stats(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get trace statistics.

        Args:
            since: Only include traces after this time.

        Returns:
            Statistics dictionary.
        """
        stats = self.storage.get_stats(since)
        return stats.to_dict()

    def get_errors(self, limit: int = 10) -> List[Trace]:
        """Get recent error traces.

        Args:
            limit: Maximum traces to return.

        Returns:
            List of error traces.
        """
        if hasattr(self.storage, 'get_errors'):
            return self.storage.get_errors(limit)
        from backend.llm.tracing.storage import TraceQuery
        return self.storage.query(TraceQuery(has_error=True, limit=limit))

    def clear(self) -> None:
        """Clear all stored traces."""
        self.storage.clear()


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer(name: str = "default") -> Tracer:
    """Get a tracer instance.

    For the default name, returns the global tracer.
    For other names, creates a new tracer.

    Args:
        name: Tracer name.

    Returns:
        Tracer instance.
    """
    if name == "default":
        return get_global_tracer()
    return Tracer(name=name)


def get_global_tracer() -> Tracer:
    """Get the global tracer instance.

    Returns:
        The global Tracer.
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(name="global")
    return _global_tracer


def set_global_tracer(tracer: Tracer) -> None:
    """Set the global tracer instance.

    Args:
        tracer: The tracer to use globally.
    """
    global _global_tracer
    _global_tracer = tracer
