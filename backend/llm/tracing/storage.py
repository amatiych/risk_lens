"""Trace storage for persisting and querying traces.

This module provides storage backends for traces, with an in-memory
implementation for development and the ability to extend to persistent
storage (database, file, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import deque
import json
import threading

from backend.llm.tracing.core import Trace, SpanStatus, SpanKind


@dataclass
class TraceQuery:
    """Query parameters for filtering traces.

    Attributes:
        name: Filter by trace name (substring match).
        status: Filter by status.
        has_error: Filter to only error traces.
        min_duration_ms: Minimum duration filter.
        max_duration_ms: Maximum duration filter.
        since: Only traces after this time.
        until: Only traces before this time.
        limit: Maximum results to return.
    """
    name: Optional[str] = None
    status: Optional[SpanStatus] = None
    has_error: Optional[bool] = None
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    limit: int = 100


@dataclass
class TraceStats:
    """Aggregated statistics across traces.

    Attributes:
        total_traces: Total number of traces.
        error_traces: Number of traces with errors.
        total_llm_calls: Total LLM API calls.
        total_tool_calls: Total tool executions.
        total_input_tokens: Aggregate input tokens.
        total_output_tokens: Aggregate output tokens.
        total_cached_tokens: Aggregate cached tokens.
        avg_duration_ms: Average trace duration.
        p50_duration_ms: Median duration.
        p95_duration_ms: 95th percentile duration.
        p99_duration_ms: 99th percentile duration.
    """
    total_traces: int = 0
    error_traces: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        if self.total_traces == 0:
            return 0.0
        return (self.error_traces / self.total_traces) * 100

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total_input = self.total_input_tokens
        if total_input == 0:
            return 0.0
        return (self.total_cached_tokens / total_input) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_traces": self.total_traces,
            "error_traces": self.error_traces,
            "error_rate": f"{self.error_rate:.1f}%",
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_hit_rate": f"{self.cache_hit_rate:.1f}%",
            "avg_duration_ms": f"{self.avg_duration_ms:.1f}",
            "p50_duration_ms": f"{self.p50_duration_ms:.1f}",
            "p95_duration_ms": f"{self.p95_duration_ms:.1f}",
            "p99_duration_ms": f"{self.p99_duration_ms:.1f}",
        }


class TraceStorage(ABC):
    """Abstract base class for trace storage backends."""

    @abstractmethod
    def save(self, trace: Trace) -> None:
        """Save a trace to storage.

        Args:
            trace: The trace to save.
        """
        pass

    @abstractmethod
    def get(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID.

        Args:
            trace_id: The trace ID.

        Returns:
            The trace if found, None otherwise.
        """
        pass

    @abstractmethod
    def query(self, query: TraceQuery) -> List[Trace]:
        """Query traces with filters.

        Args:
            query: Query parameters.

        Returns:
            List of matching traces.
        """
        pass

    @abstractmethod
    def get_recent(self, limit: int = 10) -> List[Trace]:
        """Get most recent traces.

        Args:
            limit: Maximum number to return.

        Returns:
            List of recent traces.
        """
        pass

    @abstractmethod
    def get_stats(self, since: Optional[datetime] = None) -> TraceStats:
        """Get aggregated statistics.

        Args:
            since: Only include traces after this time.

        Returns:
            TraceStats with aggregated metrics.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored traces."""
        pass


class InMemoryStorage(TraceStorage):
    """In-memory trace storage with configurable capacity.

    Uses a ring buffer to store traces with automatic eviction
    of old traces when capacity is reached.

    Attributes:
        max_traces: Maximum number of traces to store.
    """

    def __init__(self, max_traces: int = 1000):
        """Initialize storage.

        Args:
            max_traces: Maximum traces to keep in memory.
        """
        self.max_traces = max_traces
        self._traces: deque = deque(maxlen=max_traces)
        self._index: Dict[str, Trace] = {}
        self._lock = threading.Lock()

    def save(self, trace: Trace) -> None:
        """Save a trace to memory."""
        with self._lock:
            # If at capacity, remove oldest from index
            if len(self._traces) >= self.max_traces:
                oldest = self._traces[0]
                self._index.pop(oldest.trace_id, None)

            self._traces.append(trace)
            self._index[trace.trace_id] = trace

    def get(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        with self._lock:
            return self._index.get(trace_id)

    def query(self, query: TraceQuery) -> List[Trace]:
        """Query traces with filters."""
        with self._lock:
            results = []

            for trace in reversed(self._traces):
                # Apply filters
                if query.name and query.name.lower() not in trace.name.lower():
                    continue
                if query.status and trace.status != query.status:
                    continue
                if query.has_error is not None and trace.has_error != query.has_error:
                    continue
                if query.min_duration_ms and (trace.duration_ms or 0) < query.min_duration_ms:
                    continue
                if query.max_duration_ms and (trace.duration_ms or float('inf')) > query.max_duration_ms:
                    continue
                if query.since and trace.start_time < query.since:
                    continue
                if query.until and trace.start_time > query.until:
                    continue

                results.append(trace)

                if len(results) >= query.limit:
                    break

            return results

    def get_recent(self, limit: int = 10) -> List[Trace]:
        """Get most recent traces."""
        with self._lock:
            return list(reversed(list(self._traces)))[:limit]

    def get_stats(self, since: Optional[datetime] = None) -> TraceStats:
        """Get aggregated statistics."""
        with self._lock:
            traces = list(self._traces)

            if since:
                traces = [t for t in traces if t.start_time >= since]

            if not traces:
                return TraceStats()

            # Collect metrics
            durations = []
            total_llm = 0
            total_tool = 0
            total_input = 0
            total_output = 0
            total_cached = 0
            errors = 0

            for trace in traces:
                if trace.duration_ms:
                    durations.append(trace.duration_ms)
                if trace.has_error:
                    errors += 1

                total_llm += len(trace.llm_spans)
                total_tool += len(trace.tool_spans)

                tokens = trace.total_tokens
                total_input += tokens.input_tokens
                total_output += tokens.output_tokens
                total_cached += tokens.cache_read_tokens

            # Calculate percentiles
            durations.sort()
            n = len(durations)

            def percentile(p: float) -> float:
                if n == 0:
                    return 0.0
                idx = int(n * p / 100)
                return durations[min(idx, n - 1)]

            return TraceStats(
                total_traces=len(traces),
                error_traces=errors,
                total_llm_calls=total_llm,
                total_tool_calls=total_tool,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_cached_tokens=total_cached,
                avg_duration_ms=sum(durations) / len(durations) if durations else 0,
                p50_duration_ms=percentile(50),
                p95_duration_ms=percentile(95),
                p99_duration_ms=percentile(99),
            )

    def get_errors(self, limit: int = 10) -> List[Trace]:
        """Get recent error traces.

        Args:
            limit: Maximum number to return.

        Returns:
            List of error traces.
        """
        return self.query(TraceQuery(has_error=True, limit=limit))

    def get_slow_traces(self, threshold_ms: float, limit: int = 10) -> List[Trace]:
        """Get traces slower than threshold.

        Args:
            threshold_ms: Minimum duration to include.
            limit: Maximum number to return.

        Returns:
            List of slow traces.
        """
        return self.query(TraceQuery(min_duration_ms=threshold_ms, limit=limit))

    def clear(self) -> None:
        """Clear all traces."""
        with self._lock:
            self._traces.clear()
            self._index.clear()

    def export_json(self, limit: Optional[int] = None) -> str:
        """Export traces as JSON.

        Args:
            limit: Maximum traces to export.

        Returns:
            JSON string with traces.
        """
        traces = self.get_recent(limit or self.max_traces)
        return json.dumps([t.to_dict() for t in traces], indent=2)


# Global storage instance
_global_storage: Optional[TraceStorage] = None


def get_trace_storage() -> TraceStorage:
    """Get the global trace storage instance.

    Returns:
        The global TraceStorage.
    """
    global _global_storage
    if _global_storage is None:
        _global_storage = InMemoryStorage()
    return _global_storage


def set_trace_storage(storage: TraceStorage) -> None:
    """Set the global trace storage instance.

    Args:
        storage: The storage to use globally.
    """
    global _global_storage
    _global_storage = storage
