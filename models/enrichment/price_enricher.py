from abc import ABC, abstractmethod
import yfinance as yf
from models.portfolio import Portfolio
from typing import Dict,List


class PriceEnricher(ABC):

    @abstractmethod
    def get_prices(self,tickers:List[str]) -> Dict[str,float]:
        pass

    def enrich_portfolio(self, portfolio:Portfolio):
        tickers = list(portfolio.holdings.index.values)
        prices = self.get_prices(tickers)
        for ticker in tickers:
            portfolio.holdings.loc[ticker,'price'] = prices[ticker]
        portfolio.holdings['market_value'] =  portfolio.holdings['price'] * portfolio.holdings['shares']

        if portfolio.nav == 1:
            tot_mv = portfolio.holdings['market_value'].sum()
        else:
            tot_mv = portfolio.nav
        portfolio.holdings['weight'] = portfolio.holdings['market_value'] / tot_mv
        portfolio.W =  portfolio.holdings['weight'].values


class YahooFinancePriceEnricher(PriceEnricher):

    def get_prices(self, tickers):

        data = yf.download(tickers, period='1d')
        data.fillna(method='ffill', inplace=True)

        close = data['Close'].sum()
        res = {}
        for ticker in tickers:
            res[ticker] = close[ticker]
        return res

