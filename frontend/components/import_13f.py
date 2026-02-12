"""13F Filing Import UI component for Risk Lens.

Provides Streamlit interface for importing portfolios from SEC 13F filings.
"""

import io
from typing import Optional

import pandas as pd
import streamlit as st

from backend.llm.agent_13f import Agent13F, Agent13FResult
from backend.llm.providers import LLMConfig


def render_13f_import_section() -> Optional[Agent13FResult]:
    """Render the 13F filing import interface.

    Returns:
        Agent13FResult if import was successful, None otherwise.
    """
    st.header("Import from SEC 13F Filing")

    st.markdown(
        """
    Search for institutional investors by name and import their latest 13F holdings
    for risk analysis.

    **Example institutional investors:**
    - Berkshire Hathaway
    - Renaissance Technologies
    - Bridgewater Associates
    - Citadel Advisors
    - Two Sigma Investments
    """
    )

    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        company_name = st.text_input(
            "Institutional Investor Name",
            placeholder="e.g., Berkshire Hathaway",
            help="Enter the name of an institutional investment manager",
        )
    with col2:
        top_n = st.number_input(
            "Max Holdings",
            min_value=10,
            max_value=500,
            value=50,
            help="Limit to top N holdings by value (reduces processing time)",
        )

    # Info box
    with st.expander("About 13F Filings"):
        st.markdown(
            """
        **What is a 13F filing?**

        SEC Form 13F is a quarterly report filed by institutional investment managers
        with at least $100 million in assets under management. It discloses their
        U.S. equity holdings.

        **Important notes:**
        - 13F filings are filed within 45 days after the end of each quarter
        - Only long positions are reported (no shorts)
        - Holdings are identified by CUSIP, which we convert to ticker symbols
        - Some securities may not map to ticker symbols (derivatives, private placements)
        """
        )

    # Run import
    if st.button(
        "Search & Import", type="primary", disabled=not company_name
    ):
        return _run_13f_import(company_name, top_n)

    return None


def _run_13f_import(
    company_name: str, top_n: int
) -> Optional[Agent13FResult]:
    """Execute the 13F import workflow with progress display.

    Args:
        company_name: Name of the institutional investor.
        top_n: Maximum number of holdings to import.

    Returns:
        Agent13FResult if successful, None otherwise.
    """
    status_container = st.empty()
    progress_container = st.empty()

    # Get LLM provider from session state
    provider_name = st.session_state.get("llm_provider", "claude")
    config = LLMConfig(provider=provider_name)

    agent = Agent13F(config=config)

    result: Optional[Agent13FResult] = None
    status_messages = []

    with st.spinner("Processing 13F filing..."):
        # Run the agent and collect status updates
        gen = agent.run(
            f"Find and analyze the latest 13F filing for {company_name}",
            top_n=top_n,
        )

        try:
            while True:
                status = next(gen)
                status_messages.append(status)
                # Update status display
                status_container.info(status)
        except StopIteration as e:
            result = e.value

    # Clear status
    status_container.empty()

    if result and result.success:
        st.success(
            f"Successfully imported {result.mapped_tickers} positions from "
            f"{result.portfolio_name or 'portfolio'}"
        )

        # Show filing details
        _render_filing_details(result)

        # Show portfolio preview
        _render_portfolio_preview(result)

        # Show unmapped CUSIPs if any
        if result.unmapped_cusips:
            _render_unmapped_cusips(result)

        return result

    else:
        error_msg = result.error_message if result else "Import failed"
        st.error(f"Import failed: {error_msg}")

        # Show status messages for debugging
        if status_messages:
            with st.expander("Workflow Log"):
                for msg in status_messages:
                    st.text(msg)

        return None


def _render_filing_details(result: Agent13FResult) -> None:
    """Render filing details in an expander.

    Args:
        result: Agent13FResult with filing information.
    """
    with st.expander("Filing Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            if result.filer_info:
                st.metric("Filer", result.filer_info.get("name", "Unknown"))
                st.caption(f"CIK: {result.filer_info.get('cik', 'N/A')}")

        with col2:
            if result.filing_info:
                st.metric(
                    "Filing Date",
                    result.filing_info.get("filing_date", "Unknown"),
                )
                st.caption(
                    f"Report Period: {result.filing_info.get('report_date', 'N/A')}"
                )

        with col3:
            st.metric("Holdings Imported", result.mapped_tickers)
            st.caption(f"Total in filing: {result.holdings_count}")


def _render_portfolio_preview(result: Agent13FResult) -> None:
    """Render portfolio preview table.

    Args:
        result: Agent13FResult with portfolio CSV.
    """
    if not result.portfolio_csv:
        return

    st.subheader("Portfolio Preview")

    df = pd.read_csv(io.StringIO(result.portfolio_csv))
    df = df.head(20)  # Show top 20

    st.dataframe(df, use_container_width=True, hide_index=True)

    if result.mapped_tickers > 20:
        st.caption(f"Showing top 20 of {result.mapped_tickers} positions")


def _render_unmapped_cusips(result: Agent13FResult) -> None:
    """Render warning about unmapped CUSIPs.

    Args:
        result: Agent13FResult with unmapped CUSIPs.
    """
    with st.expander(
        f"Unmapped Securities ({len(result.unmapped_cusips)})", expanded=False
    ):
        st.warning(
            "The following securities could not be mapped to ticker symbols. "
            "These may be derivatives, private placements, or securities not "
            "available on major exchanges."
        )
        df = pd.DataFrame({"CUSIP": result.unmapped_cusips[:20]})
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(result.unmapped_cusips) > 20:
            st.caption(
                f"Showing first 20 of {len(result.unmapped_cusips)} unmapped"
            )


def render_analysis_summary(result: Agent13FResult) -> None:
    """Render a summary of the analysis results.

    Args:
        result: Agent13FResult with analysis data.
    """
    if not result.analysis_results:
        return

    analysis = result.analysis_results

    st.subheader("Analysis Summary")

    if analysis.get("success"):
        summary = analysis.get("summary", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Positions", summary.get("positions", "N/A"))

        with col2:
            total_value = summary.get("total_value", 0)
            st.metric(
                "Portfolio Value",
                f"${total_value:,.0f}" if total_value else "N/A",
            )

        with col3:
            var_95 = summary.get("var_95")
            if var_95:
                st.metric("VaR (95%)", f"{var_95 * 100:.2f}%")
            else:
                st.metric("VaR (95%)", "N/A")

        with col4:
            var_99 = summary.get("var_99")
            if var_99:
                st.metric("VaR (99%)", f"{var_99 * 100:.2f}%")
            else:
                st.metric("VaR (99%)", "N/A")

        # Factor exposures
        factors = analysis.get("factor_exposures", {})
        if factors:
            st.markdown("**Factor Exposures:**")
            factor_df = pd.DataFrame(
                [
                    {"Factor": k, "Beta": f"{v:.3f}"}
                    for k, v in factors.items()
                ]
            )
            st.dataframe(factor_df, use_container_width=True, hide_index=True)
    else:
        st.error(
            f"Analysis incomplete: {analysis.get('error', 'Unknown error')}"
        )
