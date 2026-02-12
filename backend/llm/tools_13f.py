"""Tool definitions and handlers for 13F filing agent workflow.

This module provides callable tools for searching SEC EDGAR,
downloading 13F filings, and converting to portfolio format.
"""

import io
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.sec.edgar_client import SECEdgarClient, Holding13F
from backend.sec.cusip_mapper import CUSIPMapper


# Shared instances for tool execution
_sec_client: Optional[SECEdgarClient] = None
_cusip_mapper: Optional[CUSIPMapper] = None


def get_sec_client() -> SECEdgarClient:
    """Get or create the SEC client singleton."""
    global _sec_client
    if _sec_client is None:
        _sec_client = SECEdgarClient()
    return _sec_client


def get_cusip_mapper() -> CUSIPMapper:
    """Get or create the CUSIP mapper singleton."""
    global _cusip_mapper
    if _cusip_mapper is None:
        _cusip_mapper = CUSIPMapper()
    return _cusip_mapper


# Tool definitions for agent
TOOLS_13F = [
    {
        "name": "search_sec_filer",
        "description": (
            "Search SEC EDGAR for an institutional investor by name. "
            "Returns the CIK (Central Index Key) and company info needed "
            "to retrieve 13F filings. Use this first to find the filer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": (
                        "Name of the institutional investor "
                        "(e.g., 'Berkshire Hathaway', 'Renaissance Technologies')"
                    ),
                }
            },
            "required": ["company_name"],
        },
    },
    {
        "name": "get_latest_13f_filing",
        "description": (
            "Get the latest 13F-HR filing for an institutional investor "
            "using their CIK. Returns the filing date, report date, and "
            "accession number needed to download holdings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cik": {
                    "type": "string",
                    "description": "SEC Central Index Key (CIK) for the filer",
                }
            },
            "required": ["cik"],
        },
    },
    {
        "name": "download_13f_holdings",
        "description": (
            "Download and parse the holdings from a 13F filing's "
            "Information Table. Returns a list of positions with CUSIPs, "
            "names, share counts, and values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cik": {
                    "type": "string",
                    "description": "SEC Central Index Key (CIK) for the filer",
                },
                "accession_number": {
                    "type": "string",
                    "description": "Accession number of the 13F filing",
                },
            },
            "required": ["cik", "accession_number"],
        },
    },
    {
        "name": "convert_13f_to_portfolio",
        "description": (
            "Convert 13F holdings with CUSIPs to portfolio format with "
            "ticker symbols. Handles CUSIP-to-ticker mapping and returns "
            "a CSV-compatible portfolio ready for analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cusip": {"type": "string"},
                            "name": {"type": "string"},
                            "shares": {"type": "number"},
                            "value": {"type": "number"},
                        },
                    },
                    "description": "List of holdings from 13F filing",
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "Limit to top N holdings by value (optional, default: all)"
                    ),
                },
            },
            "required": ["holdings"],
        },
    },
    {
        "name": "run_13f_analysis",
        "description": (
            "Run a 13F-derived portfolio through the full Risk Lens "
            "analysis pipeline. Returns VaR, factor analysis, regime "
            "analysis, and generates an AI summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_csv": {
                    "type": "string",
                    "description": "CSV string with 'ticker,shares' format",
                },
                "portfolio_name": {
                    "type": "string",
                    "description": (
                        "Name for the portfolio "
                        "(e.g., 'Berkshire Hathaway Q3 2024')"
                    ),
                },
            },
            "required": ["portfolio_csv", "portfolio_name"],
        },
    },
]


def handle_search_sec_filer(company_name: str) -> Dict[str, Any]:
    """Search SEC EDGAR for a filer by name.

    Args:
        company_name: Name or partial name of the institutional investor.

    Returns:
        Dict with search results including CIKs and company names.
    """
    try:
        client = get_sec_client()
        filers = client.search_filer(company_name)

        if not filers:
            return {
                "success": False,
                "error": f"No institutional investors found matching '{company_name}'",
                "suggestion": (
                    "Try a different spelling or the full legal name. "
                    "Note: Only institutional investment managers with "
                    "$100M+ AUM are required to file 13F reports."
                ),
            }

        return {
            "success": True,
            "query": company_name,
            "results_count": len(filers),
            "filers": [
                {
                    "cik": f.cik,
                    "name": f.name,
                    "entity_type": f.entity_type,
                }
                for f in filers[:10]  # Return top 10
            ],
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
        }


def handle_get_latest_13f_filing(cik: str) -> Dict[str, Any]:
    """Get the latest 13F filing for a CIK.

    Args:
        cik: SEC Central Index Key.

    Returns:
        Dict with filing metadata.
    """
    try:
        client = get_sec_client()
        filings = client.get_13f_filings(cik, limit=5)

        if not filings:
            return {
                "success": False,
                "error": f"No 13F filings found for CIK {cik}",
                "note": (
                    "This entity may not be required to file 13F reports, "
                    "or filings may not be available yet."
                ),
            }

        latest = filings[0]
        return {
            "success": True,
            "cik": cik,
            "latest_filing": {
                "accession_number": latest.accession_number,
                "filing_date": latest.filing_date,
                "report_date": latest.report_date,
                "form_type": latest.form_type,
            },
            "recent_filings_count": len(filings),
            "all_recent_filings": [
                {
                    "accession_number": f.accession_number,
                    "filing_date": f.filing_date,
                    "report_date": f.report_date,
                    "form_type": f.form_type,
                }
                for f in filings
            ],
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to retrieve filings: {str(e)}",
        }


def handle_download_13f_holdings(
    cik: str, accession_number: str
) -> Dict[str, Any]:
    """Download and parse 13F holdings.

    Args:
        cik: SEC Central Index Key.
        accession_number: Accession number of the filing.

    Returns:
        Dict with holdings data.
    """
    try:
        client = get_sec_client()
        holdings = client.get_13f_holdings(cik, accession_number)

        if not holdings:
            return {
                "success": False,
                "error": "No holdings found in the 13F filing",
                "note": "The Information Table may be empty or in an unexpected format.",
            }

        # Sort by value (descending)
        holdings_sorted = sorted(holdings, key=lambda h: h.value, reverse=True)

        # Calculate totals
        total_value = sum(h.value for h in holdings_sorted)
        total_positions = len(holdings_sorted)

        return {
            "success": True,
            "cik": cik,
            "accession_number": accession_number,
            "total_positions": total_positions,
            "total_value_thousands": total_value,
            "total_value_display": f"${total_value / 1000:.2f}M"
            if total_value >= 1000
            else f"${total_value}K",
            "holdings": [
                {
                    "cusip": h.cusip,
                    "name": h.name,
                    "title_of_class": h.title_of_class,
                    "shares": h.shares,
                    "value": h.value,
                    "value_display": f"${h.value / 1000:.2f}M"
                    if h.value >= 1000
                    else f"${h.value}K",
                    "share_type": h.share_type,
                }
                for h in holdings_sorted
            ],
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to download holdings: {str(e)}",
        }


def handle_convert_13f_to_portfolio(
    holdings: List[Dict], top_n: Optional[int] = None
) -> Dict[str, Any]:
    """Convert 13F holdings to portfolio format.

    Args:
        holdings: List of holding dicts with cusip, name, shares, value.
        top_n: Optional limit to top N holdings by value.

    Returns:
        Dict with portfolio CSV and mapping statistics.
    """
    try:
        # Sort by value and limit if specified
        holdings_sorted = sorted(
            holdings, key=lambda h: h.get("value", 0), reverse=True
        )
        if top_n:
            holdings_sorted = holdings_sorted[:top_n]

        # Map CUSIPs to tickers
        mapper = get_cusip_mapper()
        mapped_results = mapper.map_batch(holdings_sorted)

        # Build portfolio data
        portfolio_rows = []
        successful_mappings = []
        failed_mappings = []

        for holding, mapping in mapped_results:
            if mapping.ticker:
                # Aggregate shares for same ticker
                existing = next(
                    (r for r in portfolio_rows if r["ticker"] == mapping.ticker),
                    None,
                )
                if existing:
                    existing["shares"] += holding.get("shares", 0)
                else:
                    portfolio_rows.append(
                        {
                            "ticker": mapping.ticker,
                            "shares": holding.get("shares", 0),
                        }
                    )
                successful_mappings.append(
                    {
                        "cusip": mapping.cusip,
                        "ticker": mapping.ticker,
                        "name": holding.get("name", ""),
                        "confidence": mapping.confidence,
                        "source": mapping.source,
                    }
                )
            else:
                failed_mappings.append(
                    {
                        "cusip": mapping.cusip,
                        "name": holding.get("name", ""),
                        "shares": holding.get("shares", 0),
                        "value": holding.get("value", 0),
                        "error": mapping.error,
                    }
                )

        if not portfolio_rows:
            return {
                "success": False,
                "error": "Could not map any holdings to ticker symbols",
                "failed_mappings": failed_mappings[:20],
            }

        # Create CSV string
        df = pd.DataFrame(portfolio_rows)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        return {
            "success": True,
            "portfolio_csv": csv_string,
            "statistics": {
                "total_holdings_input": len(holdings_sorted),
                "successful_mappings": len(successful_mappings),
                "failed_mappings": len(failed_mappings),
                "unique_tickers": len(portfolio_rows),
                "mapping_success_rate": len(successful_mappings)
                / len(holdings_sorted)
                if holdings_sorted
                else 0,
            },
            "mapped_holdings": successful_mappings[:50],  # Top 50 for display
            "unmapped_holdings": failed_mappings[:20],  # Show some failures
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Conversion failed: {str(e)}",
        }


def handle_run_13f_analysis(
    portfolio_csv: str, portfolio_name: str
) -> Dict[str, Any]:
    """Run portfolio through analysis pipeline.

    Args:
        portfolio_csv: CSV string with ticker,shares columns.
        portfolio_name: Name for the portfolio.

    Returns:
        Dict with analysis results summary.
    """
    try:
        # Import here to avoid circular imports
        from frontend.services.portfolio_service import (
            enrich_portfolio,
            run_analysis,
        )
        from models.portfolio import Portfolio

        # Parse CSV to DataFrame
        df = pd.read_csv(io.StringIO(portfolio_csv))

        if "ticker" not in df.columns or "shares" not in df.columns:
            return {
                "success": False,
                "error": "Invalid portfolio CSV: missing ticker or shares columns",
            }

        # Create Portfolio object
        df["ticker"] = df["ticker"].str.upper().str.strip()
        df = df.set_index("ticker")

        portfolio = Portfolio(
            portfolio_id=0,
            name=portfolio_name,
            nav=1.0,  # Will be calculated from market values
            holdings=df,
        )

        # Enrich with market data
        portfolio = enrich_portfolio(portfolio)

        # Run analysis
        results = run_analysis(portfolio)

        # Extract key metrics for summary
        var_95 = next(
            (v for v in results.var_results if abs(v.ci - 0.95) < 0.01), None
        )
        var_99 = next(
            (v for v in results.var_results if abs(v.ci - 0.99) < 0.01), None
        )

        return {
            "success": True,
            "portfolio_name": portfolio_name,
            "summary": {
                "positions": len(portfolio.holdings),
                "total_value": float(portfolio.nav),
                "var_95": float(var_95.var) if var_95 else None,
                "var_99": float(var_99.var) if var_99 else None,
                "expected_shortfall_95": float(var_95.es) if var_95 else None,
            },
            "top_holdings": [
                {
                    "ticker": ticker,
                    "weight": float(portfolio.holdings.loc[ticker, "weight"]),
                    "market_value": float(
                        portfolio.holdings.loc[ticker, "market_value"]
                    ),
                }
                for ticker in portfolio.holdings.nlargest(10, "weight").index
            ],
            "factor_exposures": {
                factor: float(beta)
                for factor, beta in zip(
                    results.factor_result.factor_names,
                    results.factor_result.betas,
                )
            }
            if results.factor_result
            else {},
            "analysis_complete": True,
            "message": (
                f"Analysis complete for {portfolio_name}. "
                f"Portfolio has {len(portfolio.holdings)} positions "
                f"with total value ${portfolio.nav:,.0f}."
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}",
        }


def execute_13f_tool(name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a 13F tool by name.

    Args:
        name: Name of the tool to execute.
        inputs: Dictionary of input parameters.

    Returns:
        Tool execution result.

    Raises:
        ValueError: If tool name is not recognized.
    """
    handlers = {
        "search_sec_filer": lambda: handle_search_sec_filer(
            inputs["company_name"]
        ),
        "get_latest_13f_filing": lambda: handle_get_latest_13f_filing(
            inputs["cik"]
        ),
        "download_13f_holdings": lambda: handle_download_13f_holdings(
            inputs["cik"], inputs["accession_number"]
        ),
        "convert_13f_to_portfolio": lambda: handle_convert_13f_to_portfolio(
            inputs["holdings"], inputs.get("top_n")
        ),
        "run_13f_analysis": lambda: handle_run_13f_analysis(
            inputs["portfolio_csv"], inputs["portfolio_name"]
        ),
    }

    if name not in handlers:
        raise ValueError(f"Unknown 13F tool: {name}")

    return handlers[name]()
