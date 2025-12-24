"""Core tracing types for LLM observability.

This module defines the fundamental types for tracing: Span and Trace.
A Trace represents a complete operation (e.g., a chat request), while
Spans represent individual steps within that operation.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import json
import traceback


class SpanStatus(Enum):
    """Status of a span execution."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanKind(Enum):
    """Type of span for categorization."""
    LLM = "llm"              # LLM API call
    TOOL = "tool"            # Tool execution
    GUARDRAIL = "guardrail"  # Guardrail check
    CACHE = "cache"          # Cache operation
    INTERNAL = "internal"    # Internal processing


@dataclass
class TokenUsage:
    """Token usage from an LLM call.

    Attributes:
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens in the response.
        cache_read_tokens: Tokens read from cache.
        cache_creation_tokens: Tokens written to cache.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Span:
    """A span represents a single operation within a trace.

    Spans can be nested to represent hierarchical operations.
    For example, a chat request trace might contain spans for:
    - Input validation
    - LLM API call
    - Tool execution
    - Output validation

    Attributes:
        span_id: Unique identifier for this span.
        trace_id: ID of the parent trace.
        parent_span_id: ID of parent span (for nesting).
        name: Human-readable name of the operation.
        kind: Type of span (LLM, TOOL, etc.).
        start_time: When the span started.
        end_time: When the span ended.
        status: Execution status.
        attributes: Key-value metadata.
        events: Timestamped events during the span.
        tokens: Token usage (for LLM spans).
        error: Error information if failed.
    """
    span_id: str
    trace_id: str
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    tokens: Optional[TokenUsage] = None
    error: Optional[Dict[str, str]] = None

    # Internal state
    _is_recording: bool = field(default=True, repr=False)

    @classmethod
    def create(
        cls,
        name: str,
        trace_id: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
    ) -> "Span":
        """Create a new span.

        Args:
            name: Name of the operation.
            trace_id: Parent trace ID.
            kind: Type of span.
            parent_span_id: Parent span for nesting.

        Returns:
            New Span instance.
        """
        return cls(
            span_id=str(uuid.uuid4())[:8],
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
        )

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def is_error(self) -> bool:
        """Whether this span has an error."""
        return self.status == SpanStatus.ERROR

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a span attribute.

        Args:
            key: Attribute name.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        if self._is_recording:
            self.attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes.

        Args:
            attributes: Dictionary of attributes.

        Returns:
            Self for chaining.
        """
        if self._is_recording:
            self.attributes.update(attributes)
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add a timestamped event.

        Args:
            name: Event name.
            attributes: Optional event attributes.

        Returns:
            Self for chaining.
        """
        if self._is_recording:
            self.events.append({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {},
            })
        return self

    def set_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> "Span":
        """Set token usage for this span.

        Args:
            input_tokens: Input tokens used.
            output_tokens: Output tokens generated.
            cache_read_tokens: Tokens from cache.
            cache_creation_tokens: Tokens cached.

        Returns:
            Self for chaining.
        """
        if self._is_recording:
            self.tokens = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            )
        return self

    def set_tokens_from_usage(self, usage: Any) -> "Span":
        """Set tokens from an API response usage object.

        Args:
            usage: Usage object from API response.

        Returns:
            Self for chaining.
        """
        if self._is_recording and usage:
            self.tokens = TokenUsage(
                input_tokens=getattr(usage, 'input_tokens', 0),
                output_tokens=getattr(usage, 'output_tokens', 0),
                cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0),
                cache_creation_tokens=getattr(usage, 'cache_creation_input_tokens', 0),
            )
        return self

    def record_exception(self, exception: Exception) -> "Span":
        """Record an exception on this span.

        Args:
            exception: The exception that occurred.

        Returns:
            Self for chaining.
        """
        if self._is_recording:
            self.status = SpanStatus.ERROR
            self.error = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }
            self.add_event("exception", {
                "type": type(exception).__name__,
                "message": str(exception),
            })
        return self

    def end(self, status: Optional[SpanStatus] = None) -> "Span":
        """End this span.

        Args:
            status: Final status. Defaults to OK if not set.

        Returns:
            Self for chaining.
        """
        self.end_time = datetime.now()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
        self._is_recording = False
        return self

    def __enter__(self) -> "Span":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_val:
            self.record_exception(exc_val)
        self.end()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "error": self.error,
        }


@dataclass
class Trace:
    """A trace represents a complete operation with multiple spans.

    A trace groups related spans together, representing the full
    lifecycle of an operation like a chat request.

    Attributes:
        trace_id: Unique identifier for this trace.
        name: Human-readable name.
        start_time: When the trace started.
        end_time: When the trace ended.
        spans: List of spans in this trace.
        attributes: Trace-level metadata.
        status: Overall trace status.
    """
    trace_id: str
    name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    spans: List[Span] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: SpanStatus = SpanStatus.UNSET

    # Internal state
    _current_span: Optional[Span] = field(default=None, repr=False)
    _is_recording: bool = field(default=True, repr=False)

    @classmethod
    def create(cls, name: str) -> "Trace":
        """Create a new trace.

        Args:
            name: Name of the trace.

        Returns:
            New Trace instance.
        """
        return cls(
            trace_id=str(uuid.uuid4())[:12],
            name=name,
        )

    @property
    def duration_ms(self) -> Optional[float]:
        """Total duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def total_tokens(self) -> TokenUsage:
        """Aggregate token usage across all spans."""
        total = TokenUsage()
        for span in self.spans:
            if span.tokens:
                total.input_tokens += span.tokens.input_tokens
                total.output_tokens += span.tokens.output_tokens
                total.cache_read_tokens += span.tokens.cache_read_tokens
                total.cache_creation_tokens += span.tokens.cache_creation_tokens
        return total

    @property
    def has_error(self) -> bool:
        """Whether any span has an error."""
        return any(span.is_error for span in self.spans)

    @property
    def llm_spans(self) -> List[Span]:
        """Get all LLM-type spans."""
        return [s for s in self.spans if s.kind == SpanKind.LLM]

    @property
    def tool_spans(self) -> List[Span]:
        """Get all tool-type spans."""
        return [s for s in self.spans if s.kind == SpanKind.TOOL]

    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Span:
        """Create and start a new span in this trace.

        Args:
            name: Span name.
            kind: Span type.

        Returns:
            New Span instance.
        """
        parent_id = self._current_span.span_id if self._current_span else None
        span = Span.create(
            name=name,
            trace_id=self.trace_id,
            kind=kind,
            parent_span_id=parent_id,
        )
        self.spans.append(span)
        self._current_span = span
        return span

    def set_attribute(self, key: str, value: Any) -> "Trace":
        """Set a trace-level attribute.

        Args:
            key: Attribute name.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        if self._is_recording:
            self.attributes[key] = value
        return self

    def end(self, status: Optional[SpanStatus] = None) -> "Trace":
        """End this trace.

        Args:
            status: Final status.

        Returns:
            Self for chaining.
        """
        self.end_time = datetime.now()
        if status:
            self.status = status
        elif self.has_error:
            self.status = SpanStatus.ERROR
        else:
            self.status = SpanStatus.OK
        self._is_recording = False
        return self

    def __enter__(self) -> "Trace":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_val:
            self.status = SpanStatus.ERROR
        self.end()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "spans": [s.to_dict() for s in self.spans],
            "total_tokens": self.total_tokens.to_dict(),
            "has_error": self.has_error,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Get a human-readable summary."""
        tokens = self.total_tokens
        status_icon = "✅" if self.status == SpanStatus.OK else "❌"
        lines = [
            f"{status_icon} Trace: {self.name} ({self.trace_id})",
            f"   Duration: {self.duration_ms:.1f}ms" if self.duration_ms else "   Duration: in progress",
            f"   Spans: {len(self.spans)} ({len(self.llm_spans)} LLM, {len(self.tool_spans)} tool)",
            f"   Tokens: {tokens.input_tokens:,} in / {tokens.output_tokens:,} out",
        ]
        if tokens.cache_read_tokens > 0:
            lines.append(f"   Cache: {tokens.cache_read_tokens:,} tokens from cache")
        if self.has_error:
            lines.append("   ⚠️ Contains errors")
        return "\n".join(lines)
