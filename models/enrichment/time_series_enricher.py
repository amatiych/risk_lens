from abc import ABC, abstractmethod
import yfinance as yf
from pandas.core.interchange.dataframe_protocol import DataFrame

from models.portfolio import Portfolio
from typing import Dict,List

class TimeSeriesEnricher(ABC):
    @abstractmethod
    def enrich_portfolio(self,portfolio:Portfolio) -> DataFrame:
        pass


class YahooTimeSeriesEnricher(TimeSeriesEnricher):

    def enrich_portfolio(self,portfolio:Portfolio) -> DataFrame:
        tickers = list(portfolio.holdings.index.values)
        data = yf.download(tickers, period='12mo')['Close']
        #data = data.pct_change(1).dropna()
        #data.set_index("Date", inplace=True)
        portfolio.time_series = data



