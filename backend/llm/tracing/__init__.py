"""LLM Observability and Tracing Module.

This module provides comprehensive tracing for LLM interactions, including:
- Request/response logging with timing
- Token usage and cost tracking
- Error capture and debugging
- Trace visualization support

Usage:
    from backend.llm.tracing import Tracer, get_tracer

    tracer = get_tracer()

    with tracer.trace("chat_request") as trace:
        with trace.span("llm_call") as span:
            response = client.messages.create(...)
            span.set_tokens(response.usage)
            span.set_response(response)

    # View traces
    traces = tracer.get_recent_traces(10)
"""

from backend.llm.tracing.core import (
    Span,
    Trace,
    SpanStatus,
    SpanKind,
)
from backend.llm.tracing.tracer import (
    Tracer,
    get_tracer,
    get_global_tracer,
    set_global_tracer,
)
from backend.llm.tracing.storage import (
    TraceStorage,
    InMemoryStorage,
    get_trace_storage,
)
from backend.llm.tracing.instrumentation import (
    traced_llm_call,
    traced_tool_call,
    traced_guardrail,
    TracedClient,
    trace_operation,
)

__all__ = [
    # Core types
    "Span",
    "Trace",
    "SpanStatus",
    "SpanKind",
    # Tracer
    "Tracer",
    "get_tracer",
    "get_global_tracer",
    "set_global_tracer",
    # Storage
    "TraceStorage",
    "InMemoryStorage",
    "get_trace_storage",
    # Instrumentation
    "traced_llm_call",
    "traced_tool_call",
    "traced_guardrail",
    "TracedClient",
    "trace_operation",
]
