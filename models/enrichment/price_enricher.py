"""Price enrichment services for fetching current market prices.

This module provides abstract and concrete implementations for enriching
portfolio holdings with current market prices, calculating market values
and portfolio weights.
"""

from abc import ABC, abstractmethod
import yfinance as yf
from models.portfolio import Portfolio
from typing import Dict, List


class PriceEnricher(ABC):
    """Abstract base class for price enrichment services.

    Implementations fetch current prices and enrich portfolio holdings
    with price, market_value, and weight columns.
    """

    @abstractmethod
    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current prices for a list of tickers.

        Args:
            tickers: List of stock ticker symbols.

        Returns:
            Dictionary mapping ticker symbols to their current prices.
        """
        pass

    def enrich_portfolio(self, portfolio: Portfolio):
        """Enrich portfolio holdings with current prices and weights.

        Fetches current prices for all holdings, calculates market values,
        and computes portfolio weights. Modifies the portfolio in place.

        Args:
            portfolio: Portfolio instance to enrich. Will have price,
                market_value, and weight columns added to holdings DataFrame,
                and W attribute set to the weight array.
        """
        tickers = list(portfolio.holdings.index.values)
        prices = self.get_prices(tickers)
        for ticker in tickers:
            portfolio.holdings.loc[ticker, 'price'] = prices[ticker]
        portfolio.holdings['market_value'] = (
            portfolio.holdings['price'] * portfolio.holdings['shares']
        )

        if portfolio.nav == 1:
            tot_mv = portfolio.holdings['market_value'].sum()
        else:
            tot_mv = portfolio.nav
        portfolio.holdings['weight'] = portfolio.holdings['market_value'] / tot_mv
        portfolio.W = portfolio.holdings['weight'].values


class YahooFinancePriceEnricher(PriceEnricher):
    """Price enricher using Yahoo Finance API.

    Fetches real-time stock prices from Yahoo Finance for portfolio
    holdings enrichment.
    """

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current prices from Yahoo Finance.

        Args:
            tickers: List of stock ticker symbols.

        Returns:
            Dictionary mapping ticker symbols to their closing prices.
        """
        data = yf.download(tickers, period='1d')
        data.fillna(method='ffill', inplace=True)

        close = data['Close'].sum()
        res = {}
        for ticker in tickers:
            res[ticker] = close[ticker]
        return res

