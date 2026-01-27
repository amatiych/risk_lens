"""Chart generation tools for portfolio analysis reports.

This module provides tools for generating charts and visualizations
that can be included in analysis reports.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from agent.tool_registry import tool, ToolRegistry


@tool(
    name="generate_holdings_chart",
    description="Generate a pie chart or bar chart showing portfolio holdings allocation.",
    parameters={
        "chart_type": {
            "type": "string",
            "description": "Type of chart: 'pie' or 'bar' (default: bar)"
        },
        "top_n": {
            "type": "integer",
            "description": "Number of top holdings to show (default: 10)"
        },
        "output_path": {
            "type": "string",
            "description": "Path to save the chart (optional, returns base64 if not provided)"
        }
    },
    required=[],
    category="charts"
)
def generate_holdings_chart(
    chart_type: str = "bar",
    top_n: int = 10,
    output_path: Optional[str] = None,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate a chart showing portfolio holdings allocation.

    Args:
        chart_type: Type of chart ('pie' or 'bar').
        top_n: Number of top holdings to display.
        output_path: Optional path to save the chart.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with chart data or base64 encoded image.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    try:
        # Get top holdings
        holdings = portfolio.holdings.nlargest(top_n, "weight")
        tickers = holdings.index.tolist()
        weights = (holdings["weight"] * 100).values

        # Add "Other" for remaining
        other_weight = (1 - holdings["weight"].sum()) * 100
        if other_weight > 0.5:
            tickers.append("Other")
            weights = np.append(weights, other_weight)

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "pie":
            colors = sns.color_palette("husl", len(tickers))
            wedges, texts, autotexts = ax.pie(
                weights,
                labels=tickers,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title(f"{portfolio.name} - Holdings Allocation")
        else:
            colors = sns.color_palette("viridis", len(tickers))
            bars = ax.barh(tickers, weights, color=colors)
            ax.set_xlabel("Weight (%)")
            ax.set_title(f"{portfolio.name} - Top {top_n} Holdings")
            ax.invert_yaxis()

            # Add value labels
            for bar, weight in zip(bars, weights):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f'{weight:.1f}%',
                    va='center'
                )

        plt.tight_layout()

        # Save or return base64
        result = _save_or_encode_figure(fig, output_path)
        plt.close(fig)

        return {
            "chart_type": chart_type,
            "holdings_shown": len(tickers),
            **result
        }
    except Exception as e:
        return {"error": f"Failed to generate holdings chart: {str(e)}"}


@tool(
    name="generate_correlation_heatmap",
    description="Generate a heatmap showing correlations between portfolio holdings.",
    parameters={
        "top_n": {
            "type": "integer",
            "description": "Number of top holdings to include (default: 10)"
        },
        "output_path": {
            "type": "string",
            "description": "Path to save the chart (optional)"
        }
    },
    required=[],
    category="charts"
)
def generate_correlation_heatmap(
    top_n: int = 10,
    output_path: Optional[str] = None,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate a correlation heatmap for portfolio holdings.

    Args:
        top_n: Number of top holdings to include.
        output_path: Optional path to save the chart.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with chart data or base64 encoded image.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]
    registry = ToolRegistry()
    cr = registry.get_context("correlation_matrix")

    if cr is None:
        return {"error": "Correlation matrix not available. Run VaR calculation first."}

    try:
        # Get top holdings
        top_holdings = portfolio.holdings.nlargest(top_n, "weight").index.tolist()
        cr_subset = cr.loc[top_holdings, top_holdings]

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(cr_subset, dtype=bool), k=1)
        sns.heatmap(
            cr_subset,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            square=True,
            linewidths=0.5
        )

        ax.set_title(f"{portfolio.name} - Correlation Matrix")
        plt.tight_layout()

        result = _save_or_encode_figure(fig, output_path)
        plt.close(fig)

        return {
            "holdings_shown": len(top_holdings),
            **result
        }
    except Exception as e:
        return {"error": f"Failed to generate correlation heatmap: {str(e)}"}


@tool(
    name="generate_factor_chart",
    description="Generate a bar chart showing portfolio factor exposures (betas).",
    parameters={
        "output_path": {
            "type": "string",
            "description": "Path to save the chart (optional)"
        }
    },
    required=[],
    category="charts"
)
def generate_factor_chart(
    output_path: Optional[str] = None,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate a chart showing factor exposures.

    Args:
        output_path: Optional path to save the chart.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with chart data or base64 encoded image.
    """
    registry = ToolRegistry()
    factor_result = registry.get_context("factor_result")

    if factor_result is None:
        return {"error": "Factor analysis not available. Run factor analysis first."}

    portfolio = _context.get("portfolio") if _context else None
    portfolio_name = portfolio.name if portfolio else "Portfolio"

    try:
        factors = list(factor_result.factors)
        betas = [float(b) for b in factor_result.betas]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if b >= 0 else 'red' for b in betas]
        bars = ax.barh(factors, betas, color=colors)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Beta (Factor Exposure)")
        ax.set_title(f"{portfolio_name} - Factor Exposures")

        # Add value labels
        for bar, beta in zip(bars, betas):
            x_pos = bar.get_width() + 0.02 if beta >= 0 else bar.get_width() - 0.02
            ha = 'left' if beta >= 0 else 'right'
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f'{beta:.3f}',
                va='center',
                ha=ha
            )

        plt.tight_layout()

        result = _save_or_encode_figure(fig, output_path)
        plt.close(fig)

        return {
            "factors": factors,
            **result
        }
    except Exception as e:
        return {"error": f"Failed to generate factor chart: {str(e)}"}


@tool(
    name="generate_regime_chart",
    description="Generate a bar chart showing portfolio performance across market regimes.",
    parameters={
        "output_path": {
            "type": "string",
            "description": "Path to save the chart (optional)"
        }
    },
    required=[],
    category="charts"
)
def generate_regime_chart(
    output_path: Optional[str] = None,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate a chart showing regime performance.

    Args:
        output_path: Optional path to save the chart.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with chart data or base64 encoded image.
    """
    registry = ToolRegistry()
    regime_analysis = registry.get_context("regime_analysis")

    if regime_analysis is None:
        return {"error": "Regime analysis not available. Run regime analysis first."}

    portfolio = _context.get("portfolio") if _context else None
    portfolio_name = portfolio.name if portfolio else "Portfolio"

    try:
        reg_stats = regime_analysis.reg_stats

        labels = reg_stats["label"].tolist()
        returns = (reg_stats["performance"] * 100).tolist()

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if r >= 0 else 'red' for r in returns]
        bars = ax.barh(labels, returns, color=colors)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Average Daily Return (%)")
        ax.set_title(f"{portfolio_name} - Performance by Market Regime")

        # Add value labels
        for bar, ret in zip(bars, returns):
            x_pos = bar.get_width() + 0.002 if ret >= 0 else bar.get_width() - 0.002
            ha = 'left' if ret >= 0 else 'right'
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f'{ret:.3f}%',
                va='center',
                ha=ha
            )

        plt.tight_layout()

        result = _save_or_encode_figure(fig, output_path)
        plt.close(fig)

        return {
            "regimes": labels,
            **result
        }
    except Exception as e:
        return {"error": f"Failed to generate regime chart: {str(e)}"}


@tool(
    name="generate_var_chart",
    description="Generate a chart showing VaR contribution by holding.",
    parameters={
        "confidence_level": {
            "type": "number",
            "description": "Confidence level to display (default: 0.95)"
        },
        "top_n": {
            "type": "integer",
            "description": "Number of top contributors to show (default: 10)"
        },
        "output_path": {
            "type": "string",
            "description": "Path to save the chart (optional)"
        }
    },
    required=[],
    category="charts"
)
def generate_var_chart(
    confidence_level: float = 0.95,
    top_n: int = 10,
    output_path: Optional[str] = None,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate a chart showing VaR contributions.

    Args:
        confidence_level: Confidence level to display.
        top_n: Number of top contributors to show.
        output_path: Optional path to save the chart.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with chart data or base64 encoded image.
    """
    registry = ToolRegistry()
    var_engine = registry.get_context("var_engine")

    if var_engine is None or _context is None or "portfolio" not in _context:
        return {"error": "VaR data not available. Run VaR calculation first."}

    portfolio = _context["portfolio"]

    try:
        # Recalculate for specified confidence level
        var_results = var_engine.calc_var(cis=[confidence_level])
        var_result = var_results[0]

        tickers = portfolio.holdings.index.tolist()
        inc_var = [abs(iv) * 100 for iv in var_result.incremental_var]

        # Sort and get top N
        sorted_data = sorted(zip(tickers, inc_var), key=lambda x: x[1], reverse=True)
        top_data = sorted_data[:top_n]
        top_tickers = [d[0] for d in top_data]
        top_ivar = [d[1] for d in top_data]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = sns.color_palette("Reds_r", len(top_tickers))
        bars = ax.barh(top_tickers, top_ivar, color=colors)

        ax.set_xlabel("Incremental VaR Contribution (%)")
        ax.set_title(f"{portfolio.name} - VaR Risk Attribution ({confidence_level:.0%} CI)")
        ax.invert_yaxis()

        # Add value labels
        for bar, ivar in zip(bars, top_ivar):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{ivar:.2f}%',
                va='center'
            )

        plt.tight_layout()

        result = _save_or_encode_figure(fig, output_path)
        plt.close(fig)

        return {
            "confidence_level": confidence_level,
            "holdings_shown": len(top_tickers),
            **result
        }
    except Exception as e:
        return {"error": f"Failed to generate VaR chart: {str(e)}"}


@tool(
    name="generate_pca_chart",
    description="Generate a chart showing PCA variance explained by components.",
    parameters={
        "num_components": {
            "type": "integer",
            "description": "Number of components to show (default: 10)"
        },
        "output_path": {
            "type": "string",
            "description": "Path to save the chart (optional)"
        }
    },
    required=[],
    category="charts"
)
def generate_pca_chart(
    num_components: int = 10,
    output_path: Optional[str] = None,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate a chart showing PCA variance explained.

    Args:
        num_components: Number of components to display.
        output_path: Optional path to save the chart.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with chart data or base64 encoded image.
    """
    registry = ToolRegistry()
    pca_result = registry.get_context("pca_result")

    if pca_result is None:
        return {"error": "PCA analysis not available. Run PCA analysis first."}

    portfolio = _context.get("portfolio") if _context else None
    portfolio_name = portfolio.name if portfolio else "Portfolio"

    try:
        var_pct = pca_result.var_pct[:num_components] * 100
        cum_var = pca_result.cum_var_pct[:num_components] * 100
        components = [f"PC{i+1}" for i in range(len(var_pct))]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart for individual variance
        bars = ax1.bar(components, var_pct, color='steelblue', alpha=0.7, label='Individual')
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Variance Explained (%)", color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        # Line chart for cumulative variance
        ax2 = ax1.twinx()
        ax2.plot(components, cum_var, 'r-o', linewidth=2, markersize=8, label='Cumulative')
        ax2.set_ylabel("Cumulative Variance (%)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')

        ax1.set_title(f"{portfolio_name} - PCA Variance Explained")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.tight_layout()

        result = _save_or_encode_figure(fig, output_path)
        plt.close(fig)

        return {
            "components_shown": len(components),
            **result
        }
    except Exception as e:
        return {"error": f"Failed to generate PCA chart: {str(e)}"}


def _save_or_encode_figure(fig, output_path: Optional[str]) -> Dict[str, Any]:
    """Save figure to file or return as base64.

    Args:
        fig: Matplotlib figure.
        output_path: Optional path to save to.

    Returns:
        Dictionary with either file path or base64 data.
    """
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return {"saved_to": str(path)}
    else:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {"image_base64": img_base64, "format": "png"}
