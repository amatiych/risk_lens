"""Time series enrichment services for fetching historical price data.

This module provides abstract and concrete implementations for enriching
portfolios with historical price time series data for risk calculations.
"""

from abc import ABC, abstractmethod
import yfinance as yf
from pandas import DataFrame

from models.enrichment.price_enricher import PortfolioEnricher
from models.portfolio import Portfolio
from models.enrichment._yf_session import load_cached_timeseries


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
    """Time series enricher using Yahoo Finance API with cache fallback."""

    def enrich_portfolio(self, portfolio: Portfolio) -> DataFrame:
        """Fetch 12 months of historical prices, falling back to cache.

        Args:
            portfolio: Portfolio instance to enrich.

        Returns:
            DataFrame with dates as index and tickers as columns.
        """
        tickers = list(portfolio.holdings.index.values)
        try:
            data = yf.download(tickers, period='12mo')['Close']
            portfolio.time_series = data
            return data
        except BaseException:
            cached = load_cached_timeseries(tickers)
            if cached is not None:
                portfolio.time_series = cached
                return cached
            raise
