"""Tool definitions and handlers for Claude AI integration.

This module provides callable tools that Claude can use during analysis
to fetch real-time data from Yahoo Finance, including stock information,
historical prices, and market news.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

import yfinance as yf


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
        "get_market_news": lambda: handle_get_market_news(inputs["ticker"])
    }

    if name not in handlers:
        raise ValueError(f"Unknown tool: {name}")

    return handlers[name]()
