from models.factor_model import FactorModel
from models.portfolio import Portfolio
from dataclasses import dataclass
from typing import List
import statsmodels.api as sm
from statistics import mean, stdev

@dataclass
class FactorResult:
    portfolio : Portfolio
    factor_model: FactorModel
    factors: List[str]
    betas: List[float]
    portfolio_vol : float
    marginal_risk : List[float]



class FactorAnalysis:

    factor_model: FactorModel

    def __init__ (self,factor_model:FactorModel):
        self.factor_model = factor_model
        self.factorCov = self.factor_model.factors.cov()
        self.CV = self.factorCov.values
        self.factors = self.factor_model.factors.columns



    def analyze(self, portfolio:Portfolio):
        all_ts = portfolio.time_series.pct_change(1)
        all_ts.dropna(inplace=True)
        port_dates = set([d.strftime("%Y-%m-%d") for d in all_ts.index])
        factor_dates = set(self.factor_model.factors.index)
        common_dates = list(factor_dates.intersection(port_dates))
        all_ts = all_ts.loc[common_dates,:]
        factor_ts = self.factor_model.factors.loc[common_dates,:]

        T = all_ts.values
        P = T @ portfolio.W
        Y = factor_ts.values
        res = sm.OLS(Y, P).fit()

        F = res.params[0]
        S = stdev(P)
        V = S ** 2

        MC = (self.CV @ F.transpose() ) / S
        rc = F * MC
        pct_risk = rc / S
        factor_res = FactorResult(portfolio,self.factor_model ,self.factors,F,S,pct_risk)
        return factor_res

