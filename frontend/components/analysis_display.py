"""Analysis display components for visualizing portfolio risk metrics.

This module provides Streamlit components for rendering various risk
analysis visualizations including VaR metrics, correlations, factor
exposures, and regime analysis.
"""

import streamlit as st
import pandas as pd

from frontend.services.portfolio_service import AnalysisResults
from frontend.utils.formatters import (
    format_var_table,
    format_marginal_var_table,
    format_holdings_table,
    format_factor_table,
    format_currency,
    format_percentage
)


def render_holdings(results: AnalysisResults):
    """Render the portfolio holdings table.

    Args:
        results: AnalysisResults containing portfolio holdings data.
    """
    st.subheader("Holdings")
    holdings_df = format_holdings_table(results.portfolio.holdings)
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)


def render_portfolio_summary(results: AnalysisResults):
    """Render portfolio summary metrics."""
    st.subheader("Portfolio Summary")

    holdings = results.portfolio.holdings
    nav = results.portfolio.nav
    net_mv = holdings['market_value'].sum()
    gross_mv = holdings['market_value'].abs().sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NAV", format_currency(nav))
    with col2:
        st.metric("Net Market Value", format_currency(net_mv))
    with col3:
        st.metric("Gross Market Value", format_currency(gross_mv))

    col4, col5 = st.columns(2)
    with col4:
        st.metric("Net Exposure", format_percentage(net_mv / nav if nav else 1))
    with col5:
        st.metric("Gross Exposure", format_percentage(gross_mv / nav if nav else 1))


def render_var_metrics(results: AnalysisResults):
    """Render VaR metrics."""
    st.subheader("Value at Risk")

    var_df = format_var_table(results.var_results)
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    st.subheader("Marginal & Incremental VaR")
    mvar_df = format_marginal_var_table(results.var_results, results.tickers)

    for var in results.var_results:
        with st.expander(f"Details for {format_percentage(var.ci)} Confidence Level"):
            ci_df = mvar_df[mvar_df['CI'] == format_percentage(var.ci)].drop(columns=['CI'])
            st.dataframe(ci_df, use_container_width=True, hide_index=True)


def render_correlation_matrix(results: AnalysisResults):
    """Render correlation heatmap."""
    st.subheader("Correlation Matrix")

    corr = results.correlation_matrix

    styled = corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)


def render_factor_analysis(results: AnalysisResults):
    """Render factor analysis results."""
    st.subheader("Factor Analysis")

    factor_result = results.factor_result
    st.markdown(f"**Model:** {factor_result.factor_model.model_name}")
    st.markdown(f"**Portfolio Volatility:** {format_percentage(factor_result.portfolio_vol)}")

    factor_df = format_factor_table(factor_result)
    st.dataframe(factor_df, use_container_width=True, hide_index=True)

    st.subheader("Factor Exposures Chart")
    chart_data = pd.DataFrame({
        'Factor': list(factor_result.factors),
        'Beta': [float(b) for b in factor_result.betas]
    })
    st.bar_chart(chart_data.set_index('Factor'))


def render_regime_analysis(results: AnalysisResults):
    """Render regime analysis results."""
    st.subheader("Regime Analysis")

    regime = results.regime_analysis
    st.markdown("**Portfolio Performance by Market Regime:**")
    st.dataframe(regime.reg_stats, use_container_width=True)


def render_ai_summary(summary: str):
    """Render AI-generated summary."""
    st.subheader("AI Analysis Summary")
    st.markdown(summary)


def render_full_analysis(results: AnalysisResults, ai_summary: str = None):
    """Render all analysis sections."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary",
        "VaR Analysis",
        "Correlations",
        "Factor Analysis",
        "Regime Analysis"
    ])

    with tab1:
        if ai_summary:
            render_ai_summary(ai_summary)
        render_portfolio_summary(results)
        render_holdings(results)

    with tab2:
        render_var_metrics(results)

    with tab3:
        render_correlation_matrix(results)

    with tab4:
        render_factor_analysis(results)

    with tab5:
        render_regime_analysis(results)
