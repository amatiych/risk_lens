"""Analytics tools for portfolio risk analysis.

This module provides tools that wrap the risk engine functionality,
making VaR, factor analysis, regime analysis, and PCA available to the agent.
"""

from typing import Any, Dict, List, Optional

from agent.tool_registry import tool, ToolRegistry
from backend.risk_engine.var.var_engine import VarEngine
from backend.risk_engine.factor_analysis import FactorAnalysis
from backend.risk_engine.regime_analysis import RegimeAnalysis
from backend.risk_engine.portfolio_pca import PortfolioPCA
from models.factor_model import FactorModel
from models.regime_model import RegimeModel


@tool(
    name="calculate_var",
    description="Calculate Value at Risk (VaR) and Expected Shortfall for the portfolio. "
                "Returns VaR at specified confidence levels, showing potential losses. "
                "Use this to understand the portfolio's downside risk.",
    parameters={
        "confidence_levels": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Confidence levels for VaR calculation (e.g., [0.95, 0.99] for 95% and 99% VaR)"
        }
    },
    required=[],
    category="analytics"
)
def calculate_var(
    confidence_levels: Optional[List[float]] = None,
    _context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Calculate Value at Risk for the portfolio.

    Args:
        confidence_levels: List of confidence levels (default [0.95, 0.99]).
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with VaR results including VaR, ES, and risk attribution.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]
    if portfolio.time_series is None or portfolio.W is None:
        return {"error": "Portfolio not enriched. Missing time series or weights."}

    cis = confidence_levels or [0.95, 0.99]

    try:
        engine = VarEngine(portfolio.time_series, portfolio.W)
        var_results = engine.calc_var(cis=cis)

        # Store correlation matrix in context for later use
        registry = ToolRegistry()
        registry.set_context("correlation_matrix", engine.CR)
        registry.set_context("var_engine", engine)

        tickers = portfolio.holdings.index.tolist()
        results = []
        for var in var_results:
            result = {
                "confidence_level": var.ci,
                "var_pct": round(var.var * 100, 4),
                "var_description": f"{var.ci:.0%} VaR: {var.var:.2%} daily loss",
                "expected_shortfall_pct": round(var.es * 100, 4),
                "es_description": f"Average loss when VaR is breached: {var.es:.2%}",
                "var_date": var.var_date.strftime("%Y-%m-%d"),
                "marginal_var": [
                    {"ticker": t, "marginal_var": round(mv * 100, 4)}
                    for t, mv in zip(tickers, var.marginal_var)
                ],
                "incremental_var": [
                    {"ticker": t, "incremental_var": round(iv * 100, 4)}
                    for t, iv in zip(tickers, var.incremental_var)
                ]
            }
            results.append(result)

        return {
            "portfolio_name": portfolio.name,
            "var_results": results,
            "interpretation": (
                f"At {cis[0]:.0%} confidence, the portfolio could lose up to "
                f"{results[0]['var_pct']:.2f}% in a single day. "
                f"The expected shortfall (average loss in worst cases) is "
                f"{results[0]['expected_shortfall_pct']:.2f}%."
            )
        }
    except Exception as e:
        return {"error": f"VaR calculation failed: {str(e)}"}


@tool(
    name="calculate_factor_exposure",
    description="Analyze portfolio exposure to systematic risk factors (market, size, value). "
                "Uses Fama-French factor model to decompose returns. "
                "Use this to understand what's driving portfolio risk.",
    parameters={
        "factor_model_name": {
            "type": "string",
            "description": "Name of the factor model to use (default: fama_french)"
        }
    },
    required=[],
    category="analytics"
)
def calculate_factor_exposure(
    factor_model_name: str = "fama_french",
    _context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Calculate portfolio factor exposures.

    Args:
        factor_model_name: Name of the factor model to use.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with factor betas and risk contributions.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]
    if portfolio.time_series is None or portfolio.W is None:
        return {"error": "Portfolio not enriched. Missing time series or weights."}

    try:
        factor_model = FactorModel.load(factor_model_name)
        analyzer = FactorAnalysis(factor_model)
        result = analyzer.analyze(portfolio)

        # Store for later use
        registry = ToolRegistry()
        registry.set_context("factor_result", result)

        factors = []
        for factor, beta, risk_pct in zip(result.factors, result.betas, result.marginal_risk):
            factors.append({
                "factor": factor,
                "beta": round(float(beta), 4),
                "risk_contribution_pct": round(float(risk_pct) * 100, 2),
                "interpretation": _interpret_factor_beta(factor, float(beta))
            })

        return {
            "portfolio_name": portfolio.name,
            "factor_model": factor_model_name,
            "portfolio_volatility": round(result.portfolio_vol * 100, 2),
            "factors": factors,
            "summary": _summarize_factor_exposure(factors)
        }
    except Exception as e:
        return {"error": f"Factor analysis failed: {str(e)}"}


def _interpret_factor_beta(factor: str, beta: float) -> str:
    """Generate interpretation for a factor beta."""
    factor_lower = factor.lower()

    if "mkt" in factor_lower or "market" in factor_lower:
        if beta > 1.2:
            return "High market sensitivity - amplifies market moves"
        elif beta > 0.8:
            return "Moderate market exposure - moves with the market"
        else:
            return "Low market sensitivity - defensive positioning"
    elif "smb" in factor_lower or "size" in factor_lower:
        if beta > 0.2:
            return "Tilted toward small-cap stocks"
        elif beta < -0.2:
            return "Tilted toward large-cap stocks"
        else:
            return "Neutral size exposure"
    elif "hml" in factor_lower or "value" in factor_lower:
        if beta > 0.2:
            return "Tilted toward value stocks"
        elif beta < -0.2:
            return "Tilted toward growth stocks"
        else:
            return "Neutral value/growth exposure"
    else:
        return f"Beta of {beta:.2f} to {factor}"


def _summarize_factor_exposure(factors: List[Dict]) -> str:
    """Generate a summary of factor exposures."""
    parts = []
    for f in factors:
        parts.append(f"{f['factor']}: {f['beta']:.2f} ({f['risk_contribution_pct']:.1f}% of risk)")
    return "Factor exposures - " + ", ".join(parts)


@tool(
    name="analyze_regime_performance",
    description="Analyze how the portfolio performs in different market regimes "
                "(bull market, bear market, high volatility, etc.). "
                "Use this to understand portfolio behavior across market conditions.",
    parameters={
        "regime_model_name": {
            "type": "string",
            "description": "Name of the regime model to use (default: main_regime_model)"
        }
    },
    required=[],
    category="analytics"
)
def analyze_regime_performance(
    regime_model_name: str = "main_regime_model",
    _context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Analyze portfolio performance across market regimes.

    Args:
        regime_model_name: Name of the regime model to use.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with regime-specific performance statistics.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]
    if portfolio.time_series is None or portfolio.W is None:
        return {"error": "Portfolio not enriched. Missing time series or weights."}

    try:
        regime_model = RegimeModel.load(regime_model_name)
        analysis = RegimeAnalysis(portfolio, regime_model)

        # Store for later use
        registry = ToolRegistry()
        registry.set_context("regime_analysis", analysis)

        regimes = []
        for _, row in analysis.reg_stats.iterrows():
            regimes.append({
                "regime_id": int(row["regime"]),
                "label": row["label"],
                "description": row["description"],
                "avg_daily_return_pct": round(float(row["performance"]) * 100, 4),
                "interpretation": _interpret_regime_performance(
                    row["label"], float(row["performance"])
                )
            })

        return {
            "portfolio_name": portfolio.name,
            "regime_model": regime_model_name,
            "regime_performance": regimes,
            "summary": _summarize_regime_performance(regimes)
        }
    except Exception as e:
        return {"error": f"Regime analysis failed: {str(e)}"}


def _interpret_regime_performance(regime_label: str, avg_return: float) -> str:
    """Generate interpretation for regime performance."""
    return_pct = avg_return * 100
    if return_pct > 0.1:
        perf = "strong positive"
    elif return_pct > 0:
        perf = "slightly positive"
    elif return_pct > -0.1:
        perf = "slightly negative"
    else:
        perf = "negative"

    return f"Portfolio shows {perf} performance ({return_pct:.3f}% avg daily) in {regime_label}"


def _summarize_regime_performance(regimes: List[Dict]) -> str:
    """Generate summary of regime performance."""
    best = max(regimes, key=lambda r: r["avg_daily_return_pct"])
    worst = min(regimes, key=lambda r: r["avg_daily_return_pct"])
    return (
        f"Best performance in '{best['label']}' ({best['avg_daily_return_pct']:.3f}% daily), "
        f"worst in '{worst['label']}' ({worst['avg_daily_return_pct']:.3f}% daily)"
    )


@tool(
    name="run_pca_analysis",
    description="Run Principal Component Analysis on portfolio holdings to identify "
                "the main drivers of portfolio variance. Shows how much of the portfolio's "
                "movement is explained by each principal component.",
    parameters={},
    required=[],
    category="analytics"
)
def run_pca_analysis(_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run PCA analysis on portfolio holdings.

    Args:
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with PCA results including variance explained.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]
    if portfolio.time_series is None:
        return {"error": "Portfolio not enriched. Missing time series."}

    try:
        pca = PortfolioPCA(portfolio)

        # Store for later use
        registry = ToolRegistry()
        registry.set_context("pca_result", pca)

        components = []
        for i, (var_pct, cum_var) in enumerate(zip(pca.var_pct, pca.cum_var_pct)):
            components.append({
                "component": i + 1,
                "variance_explained_pct": round(float(var_pct) * 100, 2),
                "cumulative_variance_pct": round(float(cum_var) * 100, 2)
            })
            if cum_var > 0.95:  # Stop after 95% variance explained
                break

        # Find how many components explain 80% of variance
        n_80pct = next(
            (i + 1 for i, c in enumerate(components) if c["cumulative_variance_pct"] >= 80),
            len(components)
        )

        return {
            "portfolio_name": portfolio.name,
            "total_holdings": len(portfolio.holdings),
            "components": components[:10],  # Top 10 components
            "components_for_80pct_variance": n_80pct,
            "interpretation": (
                f"The first {n_80pct} principal components explain 80% of portfolio variance. "
                f"This suggests the portfolio's returns are driven by {n_80pct} main factors."
            )
        }
    except Exception as e:
        return {"error": f"PCA analysis failed: {str(e)}"}


@tool(
    name="get_correlation_matrix",
    description="Get the correlation matrix of portfolio holdings. "
                "Shows how holdings move together. Use this to identify "
                "diversification opportunities or concentration risks.",
    parameters={
        "top_n": {
            "type": "integer",
            "description": "Number of holdings to include (default: 10 for readability)"
        }
    },
    required=[],
    category="analytics"
)
def get_correlation_matrix(
    top_n: int = 10,
    _context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get correlation matrix for portfolio holdings.

    Args:
        top_n: Number of top holdings to include.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with correlation matrix and analysis.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    registry = ToolRegistry()
    cr = registry.get_context("correlation_matrix")

    if cr is None:
        # Calculate if not already done
        if portfolio.time_series is None or portfolio.W is None:
            return {"error": "Portfolio not enriched. Run VaR calculation first."}
        engine = VarEngine(portfolio.time_series, portfolio.W)
        cr = engine.CR
        registry.set_context("correlation_matrix", cr)

    try:
        # Get top holdings by weight
        top_holdings = portfolio.holdings.nlargest(top_n, "weight").index.tolist()
        cr_subset = cr.loc[top_holdings, top_holdings]

        # Find highest and lowest correlations
        corr_pairs = []
        for i, t1 in enumerate(top_holdings):
            for j, t2 in enumerate(top_holdings):
                if i < j:
                    corr_pairs.append({
                        "ticker1": t1,
                        "ticker2": t2,
                        "correlation": round(float(cr_subset.loc[t1, t2]), 4)
                    })

        corr_pairs.sort(key=lambda x: x["correlation"])
        lowest = corr_pairs[:3]
        highest = corr_pairs[-3:]

        # Build matrix as list of dicts for readability
        matrix = []
        for ticker in top_holdings:
            row = {"ticker": ticker}
            for t2 in top_holdings:
                row[t2] = round(float(cr_subset.loc[ticker, t2]), 3)
            matrix.append(row)

        return {
            "portfolio_name": portfolio.name,
            "holdings_included": top_holdings,
            "correlation_matrix": matrix,
            "highest_correlations": highest,
            "lowest_correlations": lowest,
            "interpretation": (
                f"Highest correlation: {highest[-1]['ticker1']}-{highest[-1]['ticker2']} "
                f"({highest[-1]['correlation']:.2f}). "
                f"Lowest correlation: {lowest[0]['ticker1']}-{lowest[0]['ticker2']} "
                f"({lowest[0]['correlation']:.2f})."
            )
        }
    except Exception as e:
        return {"error": f"Correlation analysis failed: {str(e)}"}


@tool(
    name="run_full_analysis",
    description="Run a comprehensive analysis of the portfolio including VaR, "
                "factor analysis, regime analysis, PCA, and correlation analysis. "
                "Use this for a complete portfolio risk assessment.",
    parameters={},
    required=[],
    category="analytics"
)
def run_full_analysis(_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run all analytics on the portfolio.

    Args:
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with all analysis results.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    results = {}

    # Run all analyses
    var_result = calculate_var(_context=_context)
    if "error" not in var_result:
        results["var_analysis"] = var_result

    factor_result = calculate_factor_exposure(_context=_context)
    if "error" not in factor_result:
        results["factor_analysis"] = factor_result

    regime_result = analyze_regime_performance(_context=_context)
    if "error" not in regime_result:
        results["regime_analysis"] = regime_result

    pca_result = run_pca_analysis(_context=_context)
    if "error" not in pca_result:
        results["pca_analysis"] = pca_result

    corr_result = get_correlation_matrix(_context=_context)
    if "error" not in corr_result:
        results["correlation_analysis"] = corr_result

    portfolio = _context["portfolio"]
    results["summary"] = {
        "portfolio_name": portfolio.name,
        "portfolio_id": portfolio.portfolio_id,
        "nav": portfolio.nav,
        "num_holdings": len(portfolio.holdings),
        "analyses_completed": list(results.keys())
    }

    return results
