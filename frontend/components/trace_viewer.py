"""Streamlit component for viewing LLM traces.

This module provides UI components for visualizing traces,
spans, and debugging LLM interactions.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from backend.llm.tracing import (
    Trace,
    Span,
    SpanKind,
    SpanStatus,
    get_tracer,
)


def render_trace_stats() -> None:
    """Render aggregate trace statistics."""
    tracer = get_tracer()
    stats = tracer.get_stats()

    st.subheader("📊 Trace Statistics")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Traces", stats.get("total_traces", 0))
    with cols[1]:
        st.metric("Error Rate", stats.get("error_rate", "0%"))
    with cols[2]:
        st.metric("Avg Duration", f"{stats.get('avg_duration_ms', '0')}ms")
    with cols[3]:
        st.metric("LLM Calls", stats.get("total_llm_calls", 0))

    # Second row
    cols2 = st.columns(4)
    with cols2[0]:
        st.metric("Input Tokens", f"{stats.get('total_input_tokens', 0):,}")
    with cols2[1]:
        st.metric("Output Tokens", f"{stats.get('total_output_tokens', 0):,}")
    with cols2[2]:
        st.metric("Cached Tokens", f"{stats.get('total_cached_tokens', 0):,}")
    with cols2[3]:
        st.metric("Cache Hit Rate", stats.get("cache_hit_rate", "0%"))


def render_trace_list(traces: List[Trace]) -> Optional[str]:
    """Render a list of traces with selection.

    Args:
        traces: List of traces to display.

    Returns:
        Selected trace ID or None.
    """
    if not traces:
        st.info("No traces recorded yet.")
        return None

    selected_id = None

    for trace in traces:
        status_icon = "✅" if trace.status == SpanStatus.OK else "❌"
        duration = f"{trace.duration_ms:.0f}ms" if trace.duration_ms else "..."
        tokens = trace.total_tokens

        # Create expander for each trace
        with st.expander(
            f"{status_icon} {trace.name} ({trace.trace_id}) - {duration}",
            expanded=False
        ):
            # Trace summary
            cols = st.columns([2, 1, 1, 1])
            with cols[0]:
                st.caption(f"Started: {trace.start_time.strftime('%H:%M:%S')}")
            with cols[1]:
                st.caption(f"Spans: {len(trace.spans)}")
            with cols[2]:
                st.caption(f"In: {tokens.input_tokens:,}")
            with cols[3]:
                st.caption(f"Out: {tokens.output_tokens:,}")

            # Span timeline
            render_span_timeline(trace)

            # Attributes
            if trace.attributes:
                st.markdown("**Attributes:**")
                st.json(trace.attributes)

            # Select button
            if st.button("View Details", key=f"view_{trace.trace_id}"):
                selected_id = trace.trace_id

    return selected_id


def render_span_timeline(trace: Trace) -> None:
    """Render a visual timeline of spans.

    Args:
        trace: Trace containing spans to visualize.
    """
    if not trace.spans:
        return

    st.markdown("**Span Timeline:**")

    for span in trace.spans:
        kind_icons = {
            SpanKind.LLM: "🤖",
            SpanKind.TOOL: "🔧",
            SpanKind.GUARDRAIL: "🛡️",
            SpanKind.CACHE: "💾",
            SpanKind.INTERNAL: "⚙️",
        }
        icon = kind_icons.get(span.kind, "•")
        status = "✅" if span.status == SpanStatus.OK else "❌"
        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "..."

        # Indent nested spans
        indent = "  " if span.parent_span_id else ""

        # Token info for LLM spans
        token_info = ""
        if span.tokens:
            token_info = f" ({span.tokens.input_tokens}→{span.tokens.output_tokens} tokens)"

        st.caption(f"{indent}{icon} {span.name}: {status} {duration}{token_info}")

        # Show error if present
        if span.error:
            st.error(f"Error: {span.error.get('message', 'Unknown')}")


def render_trace_detail(trace: Trace) -> None:
    """Render detailed view of a single trace.

    Args:
        trace: The trace to display.
    """
    st.subheader(f"Trace: {trace.name}")

    # Overview
    status_color = "green" if trace.status == SpanStatus.OK else "red"
    st.markdown(f"**Status:** :{status_color}[{trace.status.value}]")
    st.markdown(f"**Trace ID:** `{trace.trace_id}`")
    st.markdown(f"**Duration:** {trace.duration_ms:.1f}ms" if trace.duration_ms else "**Duration:** In progress")

    # Token summary
    tokens = trace.total_tokens
    st.markdown("### Token Usage")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Input", f"{tokens.input_tokens:,}")
    with cols[1]:
        st.metric("Output", f"{tokens.output_tokens:,}")
    with cols[2]:
        st.metric("Cached", f"{tokens.cache_read_tokens:,}")
    with cols[3]:
        cache_pct = (tokens.cache_read_tokens / tokens.input_tokens * 100) if tokens.input_tokens > 0 else 0
        st.metric("Cache %", f"{cache_pct:.0f}%")

    # Spans
    st.markdown("### Spans")

    for span in trace.spans:
        render_span_detail(span)

    # Attributes
    if trace.attributes:
        st.markdown("### Trace Attributes")
        st.json(trace.attributes)

    # Raw JSON
    with st.expander("Raw Trace JSON"):
        st.code(trace.to_json(), language="json")


def render_span_detail(span: Span) -> None:
    """Render detailed view of a span.

    Args:
        span: The span to display.
    """
    kind_icons = {
        SpanKind.LLM: "🤖 LLM",
        SpanKind.TOOL: "🔧 Tool",
        SpanKind.GUARDRAIL: "🛡️ Guardrail",
        SpanKind.CACHE: "💾 Cache",
        SpanKind.INTERNAL: "⚙️ Internal",
    }
    kind_label = kind_icons.get(span.kind, span.kind.value)
    status_icon = "✅" if span.status == SpanStatus.OK else "❌"

    with st.expander(
        f"{status_icon} {span.name} ({kind_label}) - {span.duration_ms:.1f}ms" if span.duration_ms else f"{status_icon} {span.name}",
        expanded=span.is_error
    ):
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.caption(f"ID: {span.span_id}")
        with cols[1]:
            st.caption(f"Kind: {span.kind.value}")
        with cols[2]:
            st.caption(f"Status: {span.status.value}")

        # Token usage
        if span.tokens:
            st.markdown("**Tokens:**")
            tcols = st.columns(4)
            with tcols[0]:
                st.caption(f"Input: {span.tokens.input_tokens:,}")
            with tcols[1]:
                st.caption(f"Output: {span.tokens.output_tokens:,}")
            with tcols[2]:
                st.caption(f"Cached: {span.tokens.cache_read_tokens:,}")
            with tcols[3]:
                st.caption(f"Created: {span.tokens.cache_creation_tokens:,}")

        # Attributes
        if span.attributes:
            st.markdown("**Attributes:**")
            st.json(span.attributes)

        # Events
        if span.events:
            st.markdown("**Events:**")
            for event in span.events:
                st.caption(f"• {event['name']} @ {event['timestamp']}")
                if event.get('attributes'):
                    st.json(event['attributes'])

        # Error
        if span.error:
            st.markdown("**Error:**")
            st.error(span.error.get('message', 'Unknown error'))
            if span.error.get('traceback'):
                with st.expander("Traceback"):
                    st.code(span.error['traceback'])


def render_trace_sidebar() -> None:
    """Render trace summary in the sidebar."""
    tracer = get_tracer()

    with st.sidebar:
        st.subheader("📡 Tracing")

        stats = tracer.get_stats()
        st.caption(f"Traces: {stats.get('total_traces', 0)}")
        st.caption(f"Errors: {stats.get('error_traces', 0)}")
        st.caption(f"Avg: {stats.get('avg_duration_ms', '0')}ms")

        # Quick error check
        errors = tracer.get_errors(limit=3)
        if errors:
            st.warning(f"⚠️ {len(errors)} recent errors")


def render_trace_viewer_page() -> None:
    """Render a full trace viewer page."""
    st.title("🔍 Trace Viewer")

    tracer = get_tracer()

    # Controls
    cols = st.columns([2, 1, 1])
    with cols[0]:
        filter_option = st.selectbox(
            "Filter",
            ["All Traces", "Errors Only", "Slow (>1s)"],
        )
    with cols[1]:
        limit = st.number_input("Limit", min_value=5, max_value=100, value=20)
    with cols[2]:
        if st.button("Clear Traces"):
            tracer.clear()
            st.rerun()

    # Stats
    render_trace_stats()

    st.divider()

    # Get traces based on filter
    if filter_option == "Errors Only":
        traces = tracer.get_errors(limit=limit)
    elif filter_option == "Slow (>1s)":
        from backend.llm.tracing.storage import TraceQuery
        traces = tracer.storage.query(TraceQuery(min_duration_ms=1000, limit=limit))
    else:
        traces = tracer.get_recent_traces(limit=limit)

    # Trace list
    st.subheader(f"Recent Traces ({len(traces)})")
    selected = render_trace_list(traces)

    # Detail view
    if selected:
        st.divider()
        trace = tracer.get_trace(selected)
        if trace:
            render_trace_detail(trace)


def init_tracing_state() -> None:
    """Initialize tracing-related session state."""
    if "tracing_enabled" not in st.session_state:
        st.session_state.tracing_enabled = True


def render_tracing_toggle() -> bool:
    """Render a toggle for tracing visibility.

    Returns:
        Whether tracing is enabled.
    """
    init_tracing_state()

    enabled = st.sidebar.checkbox(
        "📡 Enable Tracing",
        value=st.session_state.tracing_enabled,
        help="Record traces for LLM interactions"
    )
    st.session_state.tracing_enabled = enabled

    return enabled
