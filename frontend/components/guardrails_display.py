"""Streamlit component for displaying guardrails status and results.

This module provides UI components for visualizing guardrail checks,
warnings, and overall safety status in the Streamlit application.
"""

import streamlit as st
from typing import Dict, Any, List, Optional


def render_guardrails_badge(status: str) -> None:
    """Render a status badge for guardrails.

    Args:
        status: One of 'passed', 'warnings', or 'blocked'.
    """
    if status == "passed":
        st.success("✅ Guardrails: All Checks Passed")
    elif status == "warnings":
        st.warning("⚠️ Guardrails: Passed with Warnings")
    elif status == "blocked":
        st.error("🛑 Guardrails: Request Blocked")
    else:
        st.info("ℹ️ Guardrails: Status Unknown")


def render_guardrails_summary(summary: Dict[str, Any]) -> None:
    """Render a summary of guardrails results.

    Args:
        summary: Dictionary from get_guardrails_summary().
    """
    # Status badge
    render_guardrails_badge(summary.get("status", "unknown"))

    # Quick stats
    total = summary.get("total_checks", 0)
    warnings = len(summary.get("warnings", []))
    errors = len(summary.get("errors", []))

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Checks", total)
    with cols[1]:
        st.metric("Warnings", warnings, delta_color="inverse" if warnings > 0 else "off")
    with cols[2]:
        st.metric("Errors", errors, delta_color="inverse" if errors > 0 else "off")


def render_guardrails_details(summary: Dict[str, Any]) -> None:
    """Render detailed guardrails results in expandable sections.

    Args:
        summary: Dictionary from get_guardrails_summary().
    """
    with st.expander("🔍 View Guardrails Details", expanded=False):
        # Warnings section
        warnings = summary.get("warnings", [])
        if warnings:
            st.subheader("⚠️ Warnings")
            for w in warnings:
                st.warning(f"**{w['guard']}**: {w['message']}")

        # Errors section
        errors = summary.get("errors", [])
        if errors:
            st.subheader("❌ Errors")
            for e in errors:
                st.error(f"**{e['guard']}**: {e['message']}")

        # Check categories
        checks = summary.get("checks", {})

        st.subheader("📋 Check Results by Category")

        # Input checks
        if checks.get("input"):
            st.markdown("**Input Validation**")
            for check in checks["input"]:
                icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
                st.markdown(f"- {icon} {check['name']}")

        # Process checks
        if checks.get("process"):
            st.markdown("**Process Controls**")
            for check in checks["process"]:
                icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
                st.markdown(f"- {icon} {check['name']}")

        # Output checks
        if checks.get("output"):
            st.markdown("**Output Validation**")
            for check in checks["output"]:
                icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
                st.markdown(f"- {icon} {check['name']}")

        # Constitutional checks
        if checks.get("constitutional"):
            st.markdown("**Constitutional AI**")
            for check in checks["constitutional"]:
                icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
                st.markdown(f"- {icon} {check['name']}")


def render_guardrails_sidebar() -> None:
    """Render guardrails status in the sidebar."""
    with st.sidebar:
        st.subheader("🛡️ Guardrails Status")

        # Get session state guardrails info
        if "guardrails_report" in st.session_state and st.session_state.guardrails_report:
            report = st.session_state.guardrails_report
            status = "blocked" if report.get("blocked") else (
                "passed" if report.get("overall_passed", True) else "warnings"
            )
            render_guardrails_badge(status)

            # Show last check time
            if "guardrails_timestamp" in st.session_state:
                st.caption(f"Last checked: {st.session_state.guardrails_timestamp}")
        else:
            st.info("No guardrails checks run yet")


def render_injection_warning() -> None:
    """Render a warning about prompt injection detection."""
    st.error("""
    🚨 **Prompt Injection Detected**

    Your input appears to contain patterns that could be attempting to
    manipulate the AI system. This request has been blocked for safety.

    Please rephrase your question using normal language.
    """)


def render_rate_limit_warning() -> None:
    """Render a warning about rate limiting."""
    st.warning("""
    ⏱️ **Rate Limit Reached**

    You've made too many requests in a short period. Please wait a moment
    before trying again.

    This helps ensure fair access and prevents system overload.
    """)


def render_hallucination_warning(violations: List[Dict]) -> None:
    """Render a warning about potential hallucinations.

    Args:
        violations: List of hallucination violations detected.
    """
    st.warning("⚠️ **Potential Accuracy Issues Detected**")

    st.markdown("""
    Some claims in the response may not be fully supported by the source data.
    Please verify these points independently:
    """)

    for v in violations:
        st.markdown(f"- {v.get('claim', 'Unknown claim')}: {v.get('explanation', '')}")


def render_constitutional_critique(critique: Dict) -> None:
    """Render constitutional AI critique results.

    Args:
        critique: Dictionary with critique scores and issues.
    """
    with st.expander("🤖 AI Self-Critique Results", expanded=False):
        # Score metrics
        cols = st.columns(4)
        with cols[0]:
            accuracy = critique.get("factual_accuracy", 0)
            st.metric("Factual Accuracy", f"{accuracy:.0%}")
        with cols[1]:
            completeness = critique.get("completeness", 0)
            st.metric("Completeness", f"{completeness:.0%}")
        with cols[2]:
            uncertainty = "✅" if critique.get("appropriate_uncertainty") else "⚠️"
            st.metric("Uncertainty", uncertainty)
        with cols[3]:
            financial = "✅" if critique.get("financial_responsibility") else "⚠️"
            st.metric("Financial Safety", financial)

        # Issues if any
        issues = critique.get("issues", [])
        if issues:
            st.markdown("**Issues Identified:**")
            for issue in issues:
                st.markdown(f"- {issue}")

        # Suggestions if any
        suggestions = critique.get("suggestions", [])
        if suggestions:
            st.markdown("**Suggestions:**")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")


def render_disclaimer() -> None:
    """Render the financial disclaimer."""
    st.caption("""
    ---
    *Disclaimer: This analysis is for informational purposes only and does not
    constitute financial advice. Past performance does not guarantee future results.
    Please consult a qualified financial advisor before making investment decisions.*
    """)


def init_guardrails_state() -> None:
    """Initialize guardrails-related session state."""
    if "guardrails_enabled" not in st.session_state:
        st.session_state.guardrails_enabled = True
    if "guardrails_report" not in st.session_state:
        st.session_state.guardrails_report = None
    if "guardrails_timestamp" not in st.session_state:
        st.session_state.guardrails_timestamp = None


def render_guardrails_toggle() -> bool:
    """Render a toggle for enabling/disabling guardrails.

    Returns:
        Whether guardrails are enabled.
    """
    init_guardrails_state()

    enabled = st.sidebar.checkbox(
        "🛡️ Enable Guardrails",
        value=st.session_state.guardrails_enabled,
        help="Run safety and quality checks on AI interactions"
    )
    st.session_state.guardrails_enabled = enabled

    if not enabled:
        st.sidebar.warning("⚠️ Guardrails disabled - use caution")

    return enabled
