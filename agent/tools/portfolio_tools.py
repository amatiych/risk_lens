"""Portfolio management tools.

This module provides tools for loading, inspecting, and managing portfolios.
"""

from typing import Any, Dict, List, Optional

from agent.tool_registry import tool, ToolRegistry
from models.portfolio import Portfolio
from models.enrichment.price_enricher import YahooFinancePriceEnricher
from models.enrichment.time_series_enricher import YahooTimeSeriesEnricher


@tool(
    name="list_portfolios",
    description="List all available portfolios that can be analyzed. "
                "Returns portfolio IDs and names.",
    parameters={},
    required=[],
    category="portfolio"
)
def list_portfolios(**kwargs) -> Dict[str, Any]:
    """List all available portfolios.

    Returns:
        Dictionary containing list of portfolios.
    """
    try:
        portfolios_df = Portfolio.get_portfolios()
        portfolios = []
        for portfolio_id, row in portfolios_df.iterrows():
            portfolios.append({
                "portfolio_id": int(portfolio_id),
                "name": row["name"],
                "nav": float(row["nav"])
            })
        return {
            "portfolios": portfolios,
            "count": len(portfolios)
        }
    except Exception as e:
        return {"error": f"Failed to list portfolios: {str(e)}"}


@tool(
    name="load_portfolio",
    description="Load a portfolio by its ID and enrich it with current market data. "
                "This must be called before running any analysis. "
                "The portfolio will be set as the active portfolio for subsequent analysis.",
    parameters={
        "portfolio_id": {
            "type": "integer",
            "description": "The unique identifier of the portfolio to load"
        }
    },
    required=["portfolio_id"],
    category="portfolio"
)
def load_portfolio(portfolio_id: int, **kwargs) -> Dict[str, Any]:
    """Load and enrich a portfolio.

    Args:
        portfolio_id: The portfolio ID to load.

    Returns:
        Dictionary containing portfolio summary.
    """
    try:
        # Load portfolio
        portfolio = Portfolio.load(portfolio_id)

        # Enrich with current prices and time series
        price_enricher = YahooFinancePriceEnricher()
        ts_enricher = YahooTimeSeriesEnricher()

        price_enricher.enrich_portfolio(portfolio)
        ts_enricher.enrich_portfolio(portfolio)

        # Set enrichers for future trades
        portfolio.enrichers = [price_enricher, ts_enricher]

        # Store in context
        registry = ToolRegistry()
        registry.set_context("portfolio", portfolio)

        # Build holdings summary
        holdings = []
        for ticker, row in portfolio.holdings.iterrows():
            holdings.append({
                "ticker": ticker,
                "shares": float(row["shares"]),
                "price": round(float(row["price"]), 2),
                "market_value": round(float(row["market_value"]), 2),
                "weight_pct": round(float(row["weight"]) * 100, 2)
            })

        # Sort by weight
        holdings.sort(key=lambda x: x["weight_pct"], reverse=True)

        return {
            "portfolio_id": portfolio_id,
            "name": portfolio.name,
            "nav": portfolio.nav,
            "num_holdings": len(portfolio.holdings),
            "top_holdings": holdings[:10],
            "total_market_value": round(portfolio.holdings["market_value"].sum(), 2),
            "message": f"Portfolio '{portfolio.name}' loaded successfully. Ready for analysis."
        }
    except Exception as e:
        return {"error": f"Failed to load portfolio: {str(e)}"}


@tool(
    name="get_portfolio_holdings",
    description="Get the full list of holdings in the currently loaded portfolio, "
                "including shares, prices, and weights.",
    parameters={
        "sort_by": {
            "type": "string",
            "description": "Field to sort by: 'weight', 'market_value', or 'ticker' (default: weight)"
        }
    },
    required=[],
    category="portfolio"
)
def get_portfolio_holdings(
    sort_by: str = "weight",
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Get all holdings in the current portfolio.

    Args:
        sort_by: Field to sort by.
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary containing all holdings.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    try:
        holdings = []
        for ticker, row in portfolio.holdings.iterrows():
            holdings.append({
                "ticker": ticker,
                "shares": float(row["shares"]),
                "price": round(float(row["price"]), 2),
                "market_value": round(float(row["market_value"]), 2),
                "weight_pct": round(float(row["weight"]) * 100, 2)
            })

        # Sort
        if sort_by == "market_value":
            holdings.sort(key=lambda x: x["market_value"], reverse=True)
        elif sort_by == "ticker":
            holdings.sort(key=lambda x: x["ticker"])
        else:  # weight
            holdings.sort(key=lambda x: x["weight_pct"], reverse=True)

        # Calculate summary stats
        total_mv = sum(h["market_value"] for h in holdings)
        top_5_weight = sum(h["weight_pct"] for h in holdings[:5])

        return {
            "portfolio_name": portfolio.name,
            "nav": portfolio.nav,
            "holdings": holdings,
            "summary": {
                "num_holdings": len(holdings),
                "total_market_value": round(total_mv, 2),
                "top_5_concentration_pct": round(top_5_weight, 2),
                "largest_position": holdings[0] if holdings else None,
                "smallest_position": holdings[-1] if holdings else None
            }
        }
    except Exception as e:
        return {"error": f"Failed to get holdings: {str(e)}"}


@tool(
    name="get_sector_breakdown",
    description="Get the sector breakdown of the portfolio, showing allocation by sector.",
    parameters={},
    required=[],
    category="portfolio"
)
def get_sector_breakdown(
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Get sector breakdown of the portfolio.

    Args:
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary containing sector allocations.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    try:
        import yfinance as yf

        sector_weights = {}
        sector_holdings = {}

        for ticker, row in portfolio.holdings.iterrows():
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get("sector", "Unknown")
            except Exception:
                sector = "Unknown"

            weight = float(row["weight"])
            if sector not in sector_weights:
                sector_weights[sector] = 0
                sector_holdings[sector] = []
            sector_weights[sector] += weight
            sector_holdings[sector].append(ticker)

        sectors = []
        for sector, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
            sectors.append({
                "sector": sector,
                "weight_pct": round(weight * 100, 2),
                "num_holdings": len(sector_holdings[sector]),
                "holdings": sector_holdings[sector]
            })

        return {
            "portfolio_name": portfolio.name,
            "sectors": sectors,
            "num_sectors": len(sectors),
            "most_concentrated": sectors[0] if sectors else None
        }
    except Exception as e:
        return {"error": f"Failed to get sector breakdown: {str(e)}"}


@tool(
    name="execute_trade",
    description="Execute a trade (buy or sell) in the portfolio. "
                "Positive shares = buy, negative shares = sell. "
                "The portfolio will be re-enriched after the trade.",
    parameters={
        "ticker": {
            "type": "string",
            "description": "Stock ticker symbol to trade"
        },
        "shares": {
            "type": "number",
            "description": "Number of shares (positive to buy, negative to sell)"
        }
    },
    required=["ticker", "shares"],
    category="portfolio"
)
def execute_trade(
    ticker: str,
    shares: float,
    _context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Execute a trade in the portfolio.

    Args:
        ticker: Stock ticker to trade.
        shares: Number of shares (positive = buy, negative = sell).
        _context: Injected context containing the portfolio.

    Returns:
        Dictionary with trade confirmation.
    """
    if _context is None or "portfolio" not in _context:
        return {"error": "No portfolio loaded. Please load a portfolio first."}

    portfolio = _context["portfolio"]

    try:
        # Execute trade
        portfolio.trade(ticker, shares)

        # Get updated position
        if ticker in portfolio.holdings.index:
            position = portfolio.holdings.loc[ticker]
            return {
                "status": "success",
                "trade": {
                    "ticker": ticker,
                    "shares_traded": shares,
                    "action": "buy" if shares > 0 else "sell"
                },
                "new_position": {
                    "ticker": ticker,
                    "shares": float(position["shares"]),
                    "price": round(float(position["price"]), 2),
                    "market_value": round(float(position["market_value"]), 2),
                    "weight_pct": round(float(position["weight"]) * 100, 2)
                },
                "portfolio_nav": portfolio.nav
            }
        else:
            return {
                "status": "success",
                "trade": {
                    "ticker": ticker,
                    "shares_traded": shares,
                    "action": "closed position"
                },
                "message": f"Position in {ticker} closed"
            }
    except Exception as e:
        return {"error": f"Trade failed: {str(e)}"}
