"""Example: Adding Custom Tools to the Agent.

This example demonstrates how easy it is to add new tools to the
Portfolio Analysis Agent using the decorator-based registration system.

Usage:
    python examples/custom_tools.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import PortfolioAgent, tool
from agent.core import AgentConfig


# Example 1: Simple tool with no portfolio context
@tool(
    name="calculate_sharpe_ratio",
    description="Calculate the Sharpe ratio given returns and risk-free rate. "
                "Use this to evaluate risk-adjusted performance.",
    parameters={
        "annual_return": {
            "type": "number",
            "description": "Annual return as a decimal (e.g., 0.10 for 10%)"
        },
        "annual_volatility": {
            "type": "number",
            "description": "Annual volatility as a decimal (e.g., 0.15 for 15%)"
        },
        "risk_free_rate": {
            "type": "number",
            "description": "Risk-free rate as a decimal (default: 0.05 for 5%)"
        }
    },
    required=["annual_return", "annual_volatility"],
    category="analytics"
)
def calculate_sharpe_ratio(
    annual_return: float,
    annual_volatility: float,
    risk_free_rate: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """Calculate Sharpe ratio."""
    if annual_volatility == 0:
        return {"error": "Volatility cannot be zero"}

    sharpe = (annual_return - risk_free_rate) / annual_volatility

    return {
        "sharpe_ratio": round(sharpe, 4),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "risk_free_rate": risk_free_rate,
        "interpretation": _interpret_sharpe(sharpe)
    }


def _interpret_sharpe(sharpe: float) -> str:
    if sharpe < 0:
        return "Negative Sharpe ratio - returns below risk-free rate"
    elif sharpe < 1:
        return "Low Sharpe ratio - suboptimal risk-adjusted returns"
    elif sharpe < 2:
        return "Good Sharpe ratio - acceptable risk-adjusted returns"
    elif sharpe < 3:
        return "Very good Sharpe ratio - strong risk-adjusted returns"
    else:
        return "Excellent Sharpe ratio - exceptional risk-adjusted returns"


# Example 2: Tool that uses portfolio context
@tool(
    name="calculate_portfolio_beta",
    description="Calculate the overall beta of the portfolio relative to a benchmark.",
    parameters={
        "benchmark_ticker": {
            "type": "string",
            "description": "Benchmark ticker symbol (default: SPY for S&P 500)"
        }
    },
    required=[],
    category="analytics"
)
def calculate_portfolio_beta(
    benchmark_ticker: str = "SPY",
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Calculate portfolio beta relative to a benchmark."""
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    try:
        import yfinance as yf
        import numpy as np

        # Get benchmark data
        benchmark = yf.Ticker(benchmark_ticker)
        bench_hist = benchmark.history(period="1y")
        bench_returns = bench_hist["Close"].pct_change().dropna()

        # Calculate portfolio returns
        if portfolio.time_series is None or portfolio.W is None:
            return {"error": "Portfolio not enriched with time series data"}

        port_returns = (portfolio.time_series.pct_change().dropna().values @ portfolio.W)

        # Align dates
        common_length = min(len(bench_returns), len(port_returns))
        bench_returns = bench_returns[-common_length:]
        port_returns = port_returns[-common_length:]

        # Calculate beta using covariance method
        covariance = np.cov(port_returns, bench_returns.values)[0, 1]
        benchmark_variance = np.var(bench_returns.values)
        beta = covariance / benchmark_variance

        return {
            "portfolio_name": portfolio.name,
            "benchmark": benchmark_ticker,
            "beta": round(float(beta), 4),
            "interpretation": _interpret_beta(beta)
        }

    except Exception as e:
        return {"error": f"Failed to calculate beta: {str(e)}"}


def _interpret_beta(beta: float) -> str:
    if beta < 0:
        return "Negative beta - portfolio moves opposite to market"
    elif beta < 0.5:
        return "Low beta - defensive portfolio, less volatile than market"
    elif beta < 1:
        return "Moderate beta - slightly less volatile than market"
    elif beta < 1.5:
        return "High beta - more volatile than market"
    else:
        return "Very high beta - significantly more volatile than market"


# Example 3: Tool with complex output
@tool(
    name="get_concentration_metrics",
    description="Calculate portfolio concentration metrics including HHI and effective number of positions.",
    parameters={},
    required=[],
    category="analytics"
)
def get_concentration_metrics(
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Calculate portfolio concentration metrics."""
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    try:
        weights = portfolio.holdings["weight"].values

        # Herfindahl-Hirschman Index
        hhi = float(sum(w ** 2 for w in weights))

        # Effective number of positions (1/HHI)
        effective_n = 1 / hhi if hhi > 0 else len(weights)

        # Concentration ratios
        sorted_weights = sorted(weights, reverse=True)
        top_5_concentration = sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
        top_10_concentration = sum(sorted_weights[:10]) if len(sorted_weights) >= 10 else sum(sorted_weights)

        return {
            "portfolio_name": portfolio.name,
            "total_positions": len(weights),
            "herfindahl_index": round(hhi, 4),
            "effective_positions": round(effective_n, 2),
            "top_5_concentration_pct": round(top_5_concentration * 100, 2),
            "top_10_concentration_pct": round(top_10_concentration * 100, 2),
            "interpretation": _interpret_concentration(hhi, effective_n, len(weights))
        }

    except Exception as e:
        return {"error": f"Failed to calculate concentration: {str(e)}"}


def _interpret_concentration(hhi: float, effective_n: float, actual_n: int) -> str:
    parts = []

    if hhi > 0.25:
        parts.append("Highly concentrated portfolio")
    elif hhi > 0.15:
        parts.append("Moderately concentrated portfolio")
    else:
        parts.append("Well-diversified portfolio")

    diversity_ratio = effective_n / actual_n
    if diversity_ratio < 0.3:
        parts.append("position sizes are very uneven")
    elif diversity_ratio < 0.6:
        parts.append("position sizes are somewhat uneven")
    else:
        parts.append("position sizes are relatively balanced")

    return "; ".join(parts)


def main():
    # Create agent - custom tools are automatically registered via decorators above
    config = AgentConfig(
        provider="claude",
        verbose=True
    )
    agent = PortfolioAgent(config)

    # Verify our custom tools are registered
    print("Registered Analytics Tools:")
    print("=" * 50)
    for tool_name in agent.get_available_tools().get("analytics", []):
        print(f"  - {tool_name}")

    # Test the agent with our custom tools
    print("\n" + "=" * 50)
    print("Testing Custom Tools...")
    print("=" * 50)

    response = agent.analyze(
        portfolio_id=102,
        request="""Please analyze this portfolio using:
1. Calculate the portfolio beta relative to SPY
2. Get the concentration metrics
3. If you have the data, calculate the Sharpe ratio
4. Summarize what these metrics tell us about the portfolio""",
        generate_report=False
    )

    print("\n" + "=" * 50)
    print("RESULT:")
    print("=" * 50)
    print(response.content)


if __name__ == "__main__":
    main()
