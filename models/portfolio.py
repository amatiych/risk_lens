"""Portfolio data model and loading utilities.

This module defines the Portfolio class which represents an investment
portfolio with its holdings, weights, and time series data.
"""

from pandas import DataFrame, read_csv, concat
from dataclasses import dataclass
from typing import Optional, List, Tuple
from core.paths import PORTFOLIOS_FILE, get_holdings_file
from abc import ABC, abstractmethod



class PortfolioEnricher(ABC):
    pass
    """Abstract base class for all Portfolio Enrichers.
    """
    @abstractmethod
    def enrich_portfolio(self, portfolio):
        pass

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
    enrichers : Optional[List[PortfolioEnricher]]= None

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

    def trade(self, ticker, shares):
        if ticker in self.holdings.index:
            self.holdings.loc[ticker,'shares'] += shares
        else:
            new_df = DataFrame([{'ticker':ticker,'shares':shares}])
            new_df.set_index("ticker",inplace=True)
            self.holdings  = concat([self.holdings, new_df])
        self.enrich()

    def trade_many(self, trade_list : List[Tuple[str, float]]) -> None:

        new_list = []
        for ticker,shares in trade_list:
            if ticker in self.holdings.index:
                self.holdings.loc[ticker, 'shares'] += shares
            else:
               new_list.append((ticker,shares))
        if len(new_list) > 0:
            new_df =DataFrame(new_list,columns=['ticker','shares'])
            new_df.set_index("ticker",inplace=True)
            self.holdings = concat([self.holdings, new_df])
        self.enrich()

    def enrich(self):
        if self.enrichers is None:
            return

        for enricher in self.enrichers:
            enricher.enrich_portfolio(self)



if __name__ == "__main__":

    print("-----------------------------------------")
    all_portfolios = Portfolio.get_portfolios()
    print(all_portfolios)

    print("-----------------------------------------")
    portfolio = Portfolio.load(102)
    from enrichment.price_enricher import YahooFinancePriceEnricher, PortfolioEnricher
    from enrichment.time_series_enricher import YahooTimeSeriesEnricher

    enrichers = [YahooFinancePriceEnricher(),YahooTimeSeriesEnricher()]
    for enricher in enrichers:
        enricher.enrich_portfolio(portfolio)
    portfolio.enrichers = enrichers
    print("-----------------------------------------")
    print(portfolio.holdings)

    portfolio.trade('ETH-USD',1)

    print("-----------------------------------------")
    print(portfolio.holdings)
    portfolio.trade_many([('RIVN',20),('DOGE-USD',10000)])

    print("-----------------------------------------")
    print(portfolio.holdings)