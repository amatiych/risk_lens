from dataclasses import dataclass
from typing import List,Dict,Tuple
from pandas import read_csv, DataFrame, to_datetime
from models.portfolio import Portfolio
from models.regime_model import RegimeModel
from datetime import datetime
MARET_FILE = "s3://risk-lens/market/all_stocks.csv"
def get_market_ts(file):
    df =  read_csv(file)
    df['date'] = df['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    df.set_index('date', inplace=True)
    df= df.pct_change(1).dropna()
    return df


@dataclass
class PortfolioCandidate:
    ticker: str
    correlation: float
    regime_stats: Dict[str, float]

class PortfolioFitResult:

    candidates: List[PortfolioCandidate]


class PortfolioFitAnalyser():

    def __init__(self,portfolio:Portfolio):
        self.portfolio = portfolio
        all_ts = portfolio.time_series.pct_change(1).dropna()
        proforma = all_ts.values @ portfolio.W
        #dates = [int(d.strftime('%Y%m%d')) for d in all_ts.index]
        dates = all_ts.index
        self.proforma_ts = DataFrame(proforma, index=dates, columns=['Portfolio'])

    def calc_best_fit(self,market_ts: DataFrame, regime_model: RegimeModel, n:int = 10) -> DataFrame:

        reg_data = self.proforma_ts.merge(market_ts,how='inner',left_index=True,right_index=True)
        corr = reg_data.corr()
        port_corr = corr['Portfolio']
        port_corr = DataFrame(port_corr.drop('Portfolio').dropna()).sort_values(by='Portfolio')
        tickers = list(port_corr.index)[:n]

        regime_reg_data = reg_data.loc[:,tickers]
        regime_reg_data = regime_reg_data.merge(regime_model.regime_dates,how='inner',left_index=True,right_index=True)

        regime_stats = regime_reg_data.groupby('regime').mean()
        stats_by_ticker = regime_stats.to_dict()

        results : List[PortfolioFitResult] = []
        for ticker in tickers:
            candidate = PortfolioCandidate(ticker,float(port_corr.loc[ticker]['Portfolio']),stats_by_ticker[ticker])
            results.append(candidate)

        return results




