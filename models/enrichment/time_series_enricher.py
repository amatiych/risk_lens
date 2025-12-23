"""Time series enrichment services for fetching historical price data.

This module provides abstract and concrete implementations for enriching
portfolios with historical price time series data for risk calculations.
"""

from abc import ABC, abstractmethod
import yfinance as yf
from pandas import DataFrame

from models.enrichment.price_enricher import PortfolioEnricher
from models.portfolio import Portfolio


class TimeSeriesEnricher(PortfolioEnricher):
    """Abstract base class for time series enrichment services.

    Implementations fetch historical price data and attach it to
    portfolio instances for use in risk calculations.
    """

    @abstractmethod
    def enrich_portfolio(self, portfolio: Portfolio) -> DataFrame:
        """Enrich portfolio with historical time series data.

        Args:
            portfolio: Portfolio instance to enrich.

        Returns:
            DataFrame with historical prices (also attached to portfolio).
        """
        pass


class YahooTimeSeriesEnricher(TimeSeriesEnricher):
    """Time series enricher using Yahoo Finance API.

    Fetches 12 months of historical closing prices from Yahoo Finance
    for all holdings in a portfolio.
    """

    def enrich_portfolio(self, portfolio: Portfolio) -> DataFrame:
        """Fetch 12 months of historical prices from Yahoo Finance.

        Downloads closing prices for all portfolio holdings and attaches
        the time series to the portfolio's time_series attribute.

        Args:
            portfolio: Portfolio instance to enrich. Will have time_series
                attribute set to DataFrame of historical closing prices.

        Returns:
            DataFrame with dates as index and tickers as columns.
        """
        tickers = list(portfolio.holdings.index.values)
        data = yf.download(tickers, period='12mo')['Close']
        portfolio.time_series = data
        return data



