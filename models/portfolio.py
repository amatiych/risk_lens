"""Portfolio data model and loading utilities.

This module defines the Portfolio class which represents an investment
portfolio with its holdings, weights, and time series data.
"""

from pandas import DataFrame, read_csv
from dataclasses import dataclass
from typing import Optional, List
from core.paths import PORTFOLIOS_FILE, get_holdings_file


@dataclass
class Portfolio:
    """Represents an investment portfolio with holdings and market data.

    Attributes:
        portfolio_id: Unique identifier for the portfolio.
        name: Human-readable name of the portfolio.
        nav: Net Asset Value of the portfolio.
        holdings: DataFrame containing ticker, shares, price, market_value, weight.
        W: List of portfolio weights for each holding (set after enrichment).
        time_series: Historical price data for holdings (set after enrichment).
    """

    portfolio_id: int
    name: str
    nav: float
    holdings: DataFrame
    W: Optional[List[float]] = None
    time_series: Optional[DataFrame] = None

    @classmethod
    def get_portfolios(cls) -> DataFrame:
        """Get a DataFrame of all available portfolios.

        Returns:
            DataFrame indexed by portfolio_id with name and nav columns.
        """
        return read_csv(PORTFOLIOS_FILE).set_index("portfolio_id")

    @classmethod
    def load(cls, portfolio_id: int) -> "Portfolio":
        """Load a portfolio from disk by its ID.

        Args:
            portfolio_id: The unique identifier of the portfolio to load.

        Returns:
            Portfolio instance with holdings loaded from CSV.

        Raises:
            KeyError: If portfolio_id does not exist.
            FileNotFoundError: If holdings file is missing.
        """
        meta = read_csv(PORTFOLIOS_FILE).set_index("portfolio_id")
        row = meta.loc[portfolio_id]
        holdings = read_csv(get_holdings_file(portfolio_id)).set_index('ticker')

        if "Unnamed: 0" in holdings.columns:
            holdings = holdings.drop(columns="Unnamed: 0")

        return cls(
            portfolio_id=portfolio_id,
            name=row["name"],
            nav=row["nav"],
            holdings=holdings
        )

if __name__ == "__main__":

    print("-----------------------------------------")
    all_portfolios = Portfolio.get_portfolios()
    print(all_portfolios)

    print("-----------------------------------------")
    portfolio = Portfolio.load(100)
    from enrichment.price_enricher import YahooFinancePriceEnricher
    from enrichment.time_series_enricher import YahooTimeSeriesEnricher

    enrichers = [YahooFinancePriceEnricher(),YahooTimeSeriesEnricher()]
    for enricher in enrichers:
        enricher.enrich_portfolio(portfolio)
    print("-----------------------------------------")
    print(portfolio.holdings)

    print("-----------------------------------------")
    print(portfolio.time_series)
    portfolio.time_series.to_csv("data/time_series.csv")

    print("------------------------------------------")
    print("TICKERS")
    tickers =list(portfolio.time_series.columns)

    print(tickers)

    weights = list(portfolio.holdings.loc[tickers,'weight'].values)
    print("WEIGHTS")
    print(weights)
    from backend.risk_engine.var.var_engine import VarEngine
    var_engine = VarEngine(portfolio.time_series,weights)
    var = var_engine.calc_var()
    print(var)