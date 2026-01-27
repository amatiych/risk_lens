"""Market data tools for fetching external financial data.

This module provides tools for fetching stock information, historical prices,
and market news from external sources like Yahoo Finance.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf

from agent.tool_registry import tool, ToolRegistry
from backend.risk_engine.portfolio_fit import (
    PortfolioFitAnalyser,
    get_market_ts,
    MARET_FILE,
)
from models.regime_model import RegimeModel


@tool(
    name="lookup_stock_info",
    description="Get company information including sector, industry, business description, "
                "and key metrics for a stock ticker. Use this to understand what a company does.",
    parameters={
        "ticker": {
            "type": "string",
            "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        }
    },
    required=["ticker"],
    category="market_data"
)
def lookup_stock_info(ticker: str, **kwargs) -> Dict[str, Any]:
    """Fetch company information from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary containing company info.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        description = info.get("longBusinessSummary", "N/A")
        if len(description) > 500:
            description = description[:500] + "..."

        return {
            "ticker": ticker,
            "company_name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": description,
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "beta": info.get("beta", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A")
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@tool(
    name="get_historical_prices",
    description="Get historical price data for a ticker around a specific date. "
                "Use this to understand price movements around specific events.",
    parameters={
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
    required=["ticker", "date"],
    category="market_data"
)
def get_historical_prices(
    ticker: str,
    date: str,
    days_context: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """Fetch historical price data around a specific date.

    Args:
        ticker: Stock ticker symbol.
        date: Center date in YYYY-MM-DD format.
        days_context: Number of trading days before and after to include.

    Returns:
        Dictionary containing price data.
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
            daily_return = 0
            if idx != hist.index[0]:
                prev_close = hist["Close"].shift(1).loc[idx]
                if prev_close:
                    daily_return = round((row["Close"] / prev_close - 1) * 100, 2)

            prices.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
                "daily_return_pct": daily_return
            })

        prices = prices[-days_context * 2:] if len(prices) > days_context * 2 else prices
        period_return = round((prices[-1]["close"] / prices[0]["close"] - 1) * 100, 2) if prices else 0

        return {
            "ticker": ticker,
            "target_date": date,
            "prices": prices,
            "period_return_pct": period_return,
            "interpretation": f"{ticker} {'gained' if period_return > 0 else 'lost'} {abs(period_return):.1f}% over this period"
        }
    except Exception as e:
        return {"ticker": ticker, "date": date, "error": str(e)}


@tool(
    name="get_market_news",
    description="Get recent news headlines and articles for a stock ticker. "
                "Use this to understand market sentiment and recent events.",
    parameters={
        "ticker": {
            "type": "string",
            "description": "Stock ticker symbol"
        }
    },
    required=["ticker"],
    category="market_data"
)
def get_market_news(ticker: str, **kwargs) -> Dict[str, Any]:
    """Fetch recent news for a stock ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary containing recent news headlines.
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
            pub_time = item.get("providerPublishTime", 0)
            pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M") if pub_time else "N/A"

            news_items.append({
                "title": item.get("title", "N/A"),
                "publisher": item.get("publisher", "N/A"),
                "link": item.get("link", "N/A"),
                "published": pub_date
            })

        return {
            "ticker": ticker,
            "news": news_items,
            "news_count": len(news_items)
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@tool(
    name="find_portfolio_diversifiers",
    description="Find stocks that would be good diversifiers for the current portfolio "
                "based on lowest correlation and regime performance. Use this to suggest "
                "additions that could reduce portfolio risk.",
    parameters={
        "top_n": {
            "type": "integer",
            "description": "Number of candidate stocks to return (default: 10)"
        }
    },
    required=[],
    category="market_data"
)
def find_portfolio_diversifiers(
    top_n: int = 10,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Find stocks with lowest correlation to the portfolio for diversification.

    Args:
        top_n: Number of candidate stocks to return.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary containing candidate stocks with correlation and regime stats.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    try:
        # Load market data and regime model
        market_ts = get_market_ts(MARET_FILE)
        regime_model = RegimeModel.load("main_regime_model")

        # Run portfolio fit analysis
        analyser = PortfolioFitAnalyser(portfolio)
        candidates = analyser.calc_best_fit(market_ts, regime_model, n=top_n)

        # Build regime name lookup
        regime_name_lookup = {}
        if hasattr(regime_model.regime_info, 'regimes') and regime_model.regime_info.regimes:
            for regime_result in regime_model.regime_info.regimes:
                if regime_result.interpretation:
                    regime_name_lookup[regime_result.regime_id] = regime_result.interpretation.label

        # Format results
        candidate_list = []
        for candidate in candidates:
            # Format regime stats
            regime_performance = {}
            for regime_id, avg_return in candidate.regime_stats.items():
                regime_name = regime_name_lookup.get(regime_id, f"Regime {regime_id}")
                regime_performance[regime_name] = {
                    "avg_daily_return_pct": round(avg_return * 100, 4)
                }

            # Get stock info
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

        current_holdings = portfolio.holdings.index.tolist()

        return {
            "portfolio_name": portfolio.name,
            "current_holdings": current_holdings[:10],
            "total_positions": len(current_holdings),
            "diversification_candidates": candidate_list,
            "analysis_summary": {
                "candidates_found": len(candidate_list),
                "correlation_range": {
                    "lowest": round(min(c["correlation_to_portfolio"] for c in candidate_list), 4),
                    "highest": round(max(c["correlation_to_portfolio"] for c in candidate_list), 4),
                } if candidate_list else None,
                "methodology": "Candidates selected based on lowest correlation to portfolio returns, with regime-based performance analysis."
            }
        }
    except Exception as e:
        return {"error": f"Failed to analyze portfolio diversification: {str(e)}"}


def _describe_correlation(corr: float) -> str:
    """Provide a human-readable description of correlation strength."""
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
