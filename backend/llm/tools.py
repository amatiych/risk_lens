"""Tool definitions and handlers for Claude AI integration.

This module provides callable tools that Claude can use during analysis
to fetch real-time data from Yahoo Finance, including stock information,
historical prices, and market news. Also includes portfolio fit analysis
for finding diversification candidates.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf

from models.portfolio import Portfolio
from models.regime_model import RegimeModel
from backend.risk_engine.portfolio_fit import (
    PortfolioFitAnalyser,
    get_market_ts,
    MARET_FILE,
)


# Context holder for portfolio - set by chat_service before tool execution
_portfolio_context: Optional[Portfolio] = None


def set_portfolio_context(portfolio: Portfolio) -> None:
    """Set the current portfolio context for tool execution.

    Args:
        portfolio: The Portfolio instance to make available to tools.
    """
    global _portfolio_context
    _portfolio_context = portfolio


def get_portfolio_context() -> Optional[Portfolio]:
    """Get the current portfolio context.

    Returns:
        The current Portfolio instance or None if not set.
    """
    return _portfolio_context


# Tool definitions for Claude API
TOOLS = [
    {
        "name": "lookup_stock_info",
        "description": "Get company information including sector, industry, business description, and key metrics for a stock ticker",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_historical_prices",
        "description": "Get historical price data for a ticker around a specific date to understand price movements",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "date": {
                    "type": "string",
                    "description": "Center date in YYYY-MM-DD format"
                },
                "days_context": {
                    "type": "integer",
                    "description": "Number of trading days before and after the date to include (default: 5)"
                }
            },
            "required": ["ticker", "date"]
        }
    },
    {
        "name": "get_market_news",
        "description": "Get recent news headlines and articles for a stock ticker to understand market sentiment",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "find_portfolio_diversifiers",
        "description": "Find stocks that would be good diversifiers for the current portfolio based on lowest correlation and regime performance. Returns candidate stocks with their correlation to the portfolio and performance statistics across different market regimes (bull, bear, high volatility, etc.). Use this when the user asks about stocks to add, diversification opportunities, or recommendations for portfolio additions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of candidate stocks to return (default: 10)"
                }
            },
            "required": []
        }
    }
]


def handle_lookup_stock_info(ticker: str) -> Dict[str, Any]:
    """Fetch company information from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').

    Returns:
        Dictionary containing company info including sector, industry,
        description, market cap, and other key metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker,
            "company_name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": info.get("longBusinessSummary", "N/A")[:500] + "..."
                if len(info.get("longBusinessSummary", "")) > 500 else info.get("longBusinessSummary", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "beta": info.get("beta", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A")
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e)
        }


def handle_get_historical_prices(
    ticker: str,
    date: str,
    days_context: int = 5
) -> Dict[str, Any]:
    """Fetch historical price data around a specific date.

    Args:
        ticker: Stock ticker symbol.
        date: Center date in YYYY-MM-DD format.
        days_context: Number of trading days before and after to include.

    Returns:
        Dictionary containing price data with dates, OHLC, and volume.
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = target_date - timedelta(days=days_context * 2)
        end_date = target_date + timedelta(days=days_context * 2)

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return {
                "ticker": ticker,
                "date": date,
                "error": "No price data available for this period"
            }

        prices = []
        for idx, row in hist.iterrows():
            prices.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
                "daily_return": round((row["Close"] / hist["Close"].shift(1).loc[idx] - 1) * 100, 2)
                    if idx != hist.index[0] else 0
            })

        return {
            "ticker": ticker,
            "target_date": date,
            "prices": prices[-days_context*2:] if len(prices) > days_context*2 else prices,
            "period_return": round(
                (prices[-1]["close"] / prices[0]["close"] - 1) * 100, 2
            ) if prices else 0
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "date": date,
            "error": str(e)
        }


def handle_get_market_news(ticker: str) -> Dict[str, Any]:
    """Fetch recent news for a stock ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary containing list of recent news headlines and links.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return {
                "ticker": ticker,
                "news": [],
                "message": "No recent news available"
            }

        news_items = []
        for item in news[:10]:
            news_items.append({
                "title": item.get("title", "N/A"),
                "publisher": item.get("publisher", "N/A"),
                "link": item.get("link", "N/A"),
                "published": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else "N/A"
            })

        return {
            "ticker": ticker,
            "news": news_items
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e)
        }


def handle_find_portfolio_diversifiers(top_n: int = 10) -> Dict[str, Any]:
    """Find stocks with lowest correlation to the portfolio for diversification.

    Uses PortfolioFitAnalyser to identify candidate stocks based on:
    - Lowest correlation to the current portfolio
    - Performance statistics across different market regimes

    Args:
        top_n: Number of candidate stocks to return.

    Returns:
        Dictionary containing candidate stocks with correlation and regime stats.
    """
    try:
        portfolio = get_portfolio_context()
        if portfolio is None:
            return {
                "error": "No portfolio context available. Please load a portfolio first."
            }

        # Load market data and regime model
        market_ts = get_market_ts(MARET_FILE)
        regime_model = RegimeModel.load("main_regime_model")

        # Run portfolio fit analysis
        analyser = PortfolioFitAnalyser(portfolio)
        candidates = analyser.calc_best_fit(market_ts, regime_model, n=top_n)

        # Build regime name lookup from regime_info
        regime_name_lookup = {}
        if hasattr(regime_model.regime_info, 'regimes') and regime_model.regime_info.regimes:
            for regime_result in regime_model.regime_info.regimes:
                if regime_result.interpretation:
                    regime_name_lookup[regime_result.regime_id] = regime_result.interpretation.label

        # Format results for Claude
        candidate_list = []
        for candidate in candidates:
            # Format regime stats with readable regime names
            regime_performance = {}
            for regime_id, avg_return in candidate.regime_stats.items():
                regime_name = regime_name_lookup.get(regime_id, f"Regime {regime_id}")
                regime_performance[regime_name] = {
                    "avg_daily_return": round(avg_return * 100, 4),
                    "avg_daily_return_pct": f"{avg_return * 100:.4f}%"
                }

            # Get additional stock info for context
            try:
                stock = yf.Ticker(candidate.ticker)
                info = stock.info
                stock_info = {
                    "company_name": info.get("longName", candidate.ticker),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                }
            except Exception:
                stock_info = {
                    "company_name": candidate.ticker,
                    "sector": "N/A",
                    "industry": "N/A",
                }

            candidate_list.append({
                "ticker": candidate.ticker,
                "correlation_to_portfolio": round(candidate.correlation, 4),
                "correlation_description": _describe_correlation(candidate.correlation),
                "regime_performance": regime_performance,
                **stock_info
            })

        # Get current portfolio holdings for context
        current_holdings = portfolio.holdings.index.tolist()

        return {
            "portfolio_name": portfolio.name,
            "current_holdings": current_holdings[:10],  # Top 10 for context
            "total_positions": len(current_holdings),
            "diversification_candidates": candidate_list,
            "analysis_summary": {
                "candidates_found": len(candidate_list),
                "correlation_range": {
                    "lowest": round(min(c["correlation_to_portfolio"] for c in candidate_list), 4),
                    "highest": round(max(c["correlation_to_portfolio"] for c in candidate_list), 4),
                } if candidate_list else None,
                "methodology": "Candidates selected based on lowest correlation to portfolio pro-forma returns, with regime-based performance analysis."
            }
        }

    except Exception as e:
        return {
            "error": f"Failed to analyze portfolio diversification: {str(e)}"
        }


def _describe_correlation(corr: float) -> str:
    """Provide a human-readable description of correlation strength.

    Args:
        corr: Correlation coefficient (-1 to 1).

    Returns:
        Description string.
    """
    if corr < -0.5:
        return "Strong negative correlation (excellent diversifier)"
    elif corr < -0.2:
        return "Moderate negative correlation (good diversifier)"
    elif corr < 0.2:
        return "Low correlation (effective diversifier)"
    elif corr < 0.5:
        return "Moderate positive correlation (limited diversification benefit)"
    else:
        return "High positive correlation (poor diversifier)"


def execute_tool(name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given inputs.

    Args:
        name: Name of the tool to execute.
        inputs: Dictionary of input parameters for the tool.

    Returns:
        Tool execution result as a dictionary.

    Raises:
        ValueError: If tool name is not recognized.
    """
    handlers = {
        "lookup_stock_info": lambda: handle_lookup_stock_info(inputs["ticker"]),
        "get_historical_prices": lambda: handle_get_historical_prices(
            inputs["ticker"],
            inputs["date"],
            inputs.get("days_context", 5)
        ),
        "get_market_news": lambda: handle_get_market_news(inputs["ticker"]),
        "find_portfolio_diversifiers": lambda: handle_find_portfolio_diversifiers(
            inputs.get("top_n", 10)
        )
    }

    if name not in handlers:
        raise ValueError(f"Unknown tool: {name}")

    return handlers[name]()
