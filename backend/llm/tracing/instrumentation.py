"""Instrumentation utilities for automatic tracing.

This module provides decorators and wrappers for automatically
instrumenting LLM calls and other operations with tracing.
"""

import functools
import time
from typing import Callable, Any, Optional, Dict, List
from contextlib import contextmanager

from backend.llm.tracing.core import Span, SpanKind, SpanStatus
from backend.llm.tracing.tracer import get_tracer, Tracer


def traced_llm_call(
    name: Optional[str] = None,
    tracer: Optional[Tracer] = None,
) -> Callable:
    """Decorator to trace an LLM call function.

    Automatically creates a span for the decorated function,
    capturing timing and extracting token usage from the response.

    Args:
        name: Span name. Defaults to function name.
        tracer: Tracer to use. Uses global if None.

    Returns:
        Decorated function.

    Example:
        @traced_llm_call()
        def call_claude(messages):
            return client.messages.create(
                model="claude-sonnet-4-20250514",
                messages=messages
            )
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            _tracer = tracer or get_tracer()
            span_name = name or func.__name__

            with _tracer.span(span_name, SpanKind.LLM) as span:
                span.set_attribute("function", func.__name__)

                try:
                    result = func(*args, **kwargs)

                    # Extract usage from response if available
                    if hasattr(result, 'usage'):
                        span.set_tokens_from_usage(result.usage)

                    # Extract model from response if available
                    if hasattr(result, 'model'):
                        span.set_attribute("model", result.model)

                    # Extract stop reason if available
                    if hasattr(result, 'stop_reason'):
                        span.set_attribute("stop_reason", result.stop_reason)

                    return result

                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def traced_tool_call(
    name: Optional[str] = None,
    tracer: Optional[Tracer] = None,
) -> Callable:
    """Decorator to trace a tool execution.

    Args:
        name: Span name. Defaults to function name.
        tracer: Tracer to use. Uses global if None.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            _tracer = tracer or get_tracer()
            span_name = name or f"tool:{func.__name__}"

            with _tracer.span(span_name, SpanKind.TOOL) as span:
                span.set_attribute("tool_name", func.__name__)

                # Capture input args (limited to prevent huge spans)
                if args:
                    span.set_attribute("args_count", len(args))
                if kwargs:
                    span.set_attribute("kwargs_keys", list(kwargs.keys())[:10])

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result

                except Exception as e:
                    span.set_attribute("success", False)
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def traced_guardrail(
    name: Optional[str] = None,
    tracer: Optional[Tracer] = None,
) -> Callable:
    """Decorator to trace a guardrail check.

    Args:
        name: Span name. Defaults to function name.
        tracer: Tracer to use. Uses global if None.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            _tracer = tracer or get_tracer()
            span_name = name or f"guardrail:{func.__name__}"

            with _tracer.span(span_name, SpanKind.GUARDRAIL) as span:
                span.set_attribute("guardrail", func.__name__)

                try:
                    result = func(*args, **kwargs)

                    # If result is a GuardResult, extract info
                    if hasattr(result, 'passed'):
                        span.set_attribute("passed", result.passed)
                    if hasattr(result, 'severity'):
                        span.set_attribute("severity", result.severity)

                    return result

                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


class TracedClient:
    """Wrapper around Anthropic client with automatic tracing.

    This class wraps the Anthropic client to automatically trace
    all API calls with detailed metrics.

    Example:
        import anthropic
        client = TracedClient(anthropic.Anthropic())

        # Calls are automatically traced
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(
        self,
        client: Any,
        tracer: Optional[Tracer] = None,
    ):
        """Initialize the traced client.

        Args:
            client: The Anthropic client to wrap.
            tracer: Tracer to use. Uses global if None.
        """
        self._client = client
        self._tracer = tracer or get_tracer()
        self._messages = TracedMessages(client.messages, self._tracer)

    @property
    def messages(self) -> "TracedMessages":
        """Get the traced messages API."""
        return self._messages


class TracedMessages:
    """Traced wrapper for the messages API."""

    def __init__(self, messages_api: Any, tracer: Tracer):
        """Initialize.

        Args:
            messages_api: The messages API to wrap.
            tracer: Tracer to use.
        """
        self._api = messages_api
        self._tracer = tracer

    def create(self, **kwargs) -> Any:
        """Create a message with tracing.

        Args:
            **kwargs: Arguments passed to messages.create().

        Returns:
            API response.
        """
        model = kwargs.get("model", "unknown")
        max_tokens = kwargs.get("max_tokens", 0)

        with self._tracer.span("messages.create", SpanKind.LLM) as span:
            span.set_attributes({
                "model": model,
                "max_tokens": max_tokens,
                "has_tools": "tools" in kwargs,
                "has_system": "system" in kwargs,
            })

            # Count messages
            messages = kwargs.get("messages", [])
            span.set_attribute("message_count", len(messages))

            start_time = time.perf_counter()

            try:
                response = self._api.create(**kwargs)

                # Record timing
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("elapsed_ms", elapsed_ms)

                # Record usage
                if hasattr(response, 'usage'):
                    span.set_tokens_from_usage(response.usage)

                # Record response info
                if hasattr(response, 'stop_reason'):
                    span.set_attribute("stop_reason", response.stop_reason)
                if hasattr(response, 'model'):
                    span.set_attribute("response_model", response.model)

                return response

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("elapsed_ms", elapsed_ms)
                span.record_exception(e)
                raise

    @contextmanager
    def stream(self, **kwargs):
        """Stream a message with tracing.

        Args:
            **kwargs: Arguments passed to messages.stream().

        Yields:
            Stream context.
        """
        model = kwargs.get("model", "unknown")

        with self._tracer.span("messages.stream", SpanKind.LLM) as span:
            span.set_attributes({
                "model": model,
                "streaming": True,
            })

            start_time = time.perf_counter()

            try:
                with self._api.stream(**kwargs) as stream:
                    yield stream

                    # Get final message for usage
                    final = stream.get_final_message()
                    if final and hasattr(final, 'usage'):
                        span.set_tokens_from_usage(final.usage)
                    if final and hasattr(final, 'stop_reason'):
                        span.set_attribute("stop_reason", final.stop_reason)

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("elapsed_ms", elapsed_ms)

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("elapsed_ms", elapsed_ms)
                span.record_exception(e)
                raise


@contextmanager
def trace_operation(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    tracer: Optional[Tracer] = None,
):
    """Context manager for tracing an arbitrary operation.

    Args:
        name: Operation name.
        kind: Span kind.
        attributes: Initial attributes.
        tracer: Tracer to use.

    Yields:
        The span for the operation.

    Example:
        with trace_operation("process_data", SpanKind.INTERNAL) as span:
            span.set_attribute("item_count", 100)
            # do work
    """
    _tracer = tracer or get_tracer()

    with _tracer.span(name, kind, attributes) as span:
        yield span
