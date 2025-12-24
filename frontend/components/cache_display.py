"""Streamlit component for displaying prompt cache status and metrics.

This module provides UI components for visualizing cache performance,
helping users understand cost savings from Anthropic's prompt caching.
"""

import streamlit as st
from typing import Dict, Any, Optional

from backend.llm.caching import CacheMetrics, CacheStats


def render_cache_badge(cache_hit: bool) -> None:
    """Render a cache hit/miss badge.

    Args:
        cache_hit: Whether the last call was a cache hit.
    """
    if cache_hit:
        st.success("💾 Cache Hit")
    else:
        st.info("📝 Cache Miss (warming)")


def render_cache_metrics(metrics: CacheMetrics) -> None:
    """Render detailed metrics from a single API call.

    Args:
        metrics: CacheMetrics from the last call.
    """
    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "Input Tokens",
            f"{metrics.input_tokens:,}",
            help="Total input tokens for this call"
        )

    with cols[1]:
        st.metric(
            "Cached Tokens",
            f"{metrics.cache_read_tokens:,}",
            delta=f"{metrics.cache_hit_ratio:.0%}" if metrics.cache_hit else None,
            help="Tokens read from cache (90% cheaper)"
        )

    with cols[2]:
        st.metric(
            "Output Tokens",
            f"{metrics.output_tokens:,}",
            help="Response tokens generated"
        )

    with cols[3]:
        savings = metrics.estimated_savings_percent
        st.metric(
            "Cost Savings",
            f"{savings:.1f}%",
            delta="from caching" if savings > 0 else None,
            delta_color="normal",
            help="Estimated cost reduction from cache"
        )


def render_cache_stats(stats: CacheStats) -> None:
    """Render session-wide cache statistics.

    Args:
        stats: Aggregated CacheStats for the session.
    """
    st.subheader("📊 Session Cache Statistics")

    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "API Calls",
            stats.total_calls,
            help="Total API calls this session"
        )

    with cols[1]:
        st.metric(
            "Cache Hits",
            stats.cache_hits,
            delta=f"{stats.cache_hit_rate:.0f}%" if stats.total_calls > 0 else None,
            help="Calls that used cached context"
        )

    with cols[2]:
        st.metric(
            "Cached Tokens",
            f"{stats.total_cached_tokens:,}",
            delta=f"{stats.token_cache_rate:.0f}% of input",
            help="Total tokens served from cache"
        )

    with cols[3]:
        st.metric(
            "Est. Savings",
            f"{stats.estimated_total_savings_percent:.1f}%",
            help="Overall cost reduction from caching"
        )


def render_cache_sidebar() -> None:
    """Render cache status in the sidebar."""
    from frontend.services.chat_service import get_cache_status_display

    with st.sidebar:
        st.subheader("💾 Prompt Cache")

        try:
            status = get_cache_status_display()

            if status["enabled"]:
                session = status["session"]

                # Quick stats
                st.caption(f"Calls: {session['total_calls']} | Hit rate: {session['cache_hit_rate']}")
                st.caption(f"Cached tokens: {session['total_cached_tokens']:,}")
                st.caption(f"Est. savings: {session['estimated_savings']}")

                # Last call status
                if status["last_call"]:
                    last = status["last_call"]
                    if last["cache_hit"]:
                        st.success(f"✅ Last: Cache hit ({last['cached_tokens']:,} tokens)")
                    else:
                        st.info("📝 Last: Cache warming")
            else:
                st.info("Cache disabled")

        except Exception:
            st.caption("Cache stats unavailable")


def render_cache_details_expander(metrics: Optional[CacheMetrics] = None) -> None:
    """Render cache details in an expandable section.

    Args:
        metrics: Optional CacheMetrics to display.
    """
    from frontend.services.chat_service import get_cached_client

    with st.expander("💾 Prompt Cache Details", expanded=False):
        st.markdown("""
        **Prompt caching** reduces API costs by caching large, repeated context
        (like portfolio data). Cached tokens cost 90% less than regular tokens.
        """)

        client = get_cached_client()
        tracker = client.get_stats()
        stats = tracker.get_stats()

        if stats.total_calls > 0:
            render_cache_stats(stats)

            # Show recent calls
            recent = tracker.get_recent_metrics(5)
            if recent:
                st.markdown("**Recent Calls:**")
                for i, m in enumerate(reversed(recent)):
                    status = "✅ Hit" if m.cache_hit else "📝 Miss"
                    st.caption(
                        f"{i+1}. {status} | "
                        f"In: {m.input_tokens:,} | "
                        f"Cached: {m.cache_read_tokens:,} | "
                        f"Savings: {m.estimated_savings_percent:.1f}%"
                    )
        else:
            st.info("No API calls yet. Cache statistics will appear after first call.")

        # Explain how it works
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. First call writes system prompt + tools to cache
        2. Subsequent calls (within 5 min) read from cache
        3. Cached tokens cost 90% less ($0.30 vs $3.00 per 1M tokens)
        """)


def render_cache_savings_banner(metrics: CacheMetrics) -> None:
    """Render a banner showing cost savings from caching.

    Args:
        metrics: CacheMetrics from the last call.
    """
    if metrics.cache_hit and metrics.estimated_savings_percent > 0:
        st.success(
            f"💰 **Cache Hit!** Saved ~{metrics.estimated_savings_percent:.0f}% "
            f"on this call ({metrics.cache_read_tokens:,} cached tokens)"
        )


def init_cache_state() -> None:
    """Initialize cache-related session state."""
    if "cache_enabled" not in st.session_state:
        st.session_state.cache_enabled = True
    if "show_cache_metrics" not in st.session_state:
        st.session_state.show_cache_metrics = True


def render_cache_toggle() -> bool:
    """Render a toggle for cache display visibility.

    Returns:
        Whether to show cache metrics.
    """
    init_cache_state()

    show = st.sidebar.checkbox(
        "📊 Show Cache Metrics",
        value=st.session_state.show_cache_metrics,
        help="Display prompt cache performance metrics"
    )
    st.session_state.show_cache_metrics = show

    return show
