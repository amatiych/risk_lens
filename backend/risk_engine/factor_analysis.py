"""Multi-factor risk analysis for portfolio factor exposure decomposition.

This module provides tools for analyzing portfolio exposure to systematic
risk factors using regression-based factor models (e.g., Fama-French).
"""

from models.factor_model import FactorModel
from models.portfolio import Portfolio
from dataclasses import dataclass
from typing import List
import statsmodels.api as sm
from statistics import mean, stdev


@dataclass
class FactorResult:
    """Container for factor analysis results.

    Stores factor exposures (betas) and risk contributions for a portfolio.

    Attributes:
        portfolio: The analyzed Portfolio instance.
        factor_model: The FactorModel used for analysis.
        factors: List of factor names.
        betas: Factor exposures (regression coefficients).
        portfolio_vol: Portfolio volatility (standard deviation).
        marginal_risk: Percentage of risk attributable to each factor.
    """

    portfolio: Portfolio
    factor_model: FactorModel
    factors: List[str]
    betas: List[float]
    portfolio_vol: float
    marginal_risk: List[float]


class FactorAnalysis:
    """Performs multi-factor risk analysis on portfolios.

    Uses OLS regression to decompose portfolio returns into factor exposures
    and calculates risk attribution by factor.

    Attributes:
        factor_model: The FactorModel containing factor return data.
        factorCov: Covariance matrix of factor returns.
        CV: NumPy array of factor covariances.
        factors: List of factor names.

    Example:
        model = FactorModel.load('fama_french')
        analyzer = FactorAnalysis(model)
        result = analyzer.analyze(portfolio)
        print(f"Market Beta: {result.betas[0]:.2f}")
    """

    factor_model: FactorModel

    def __init__(self, factor_model: FactorModel):
        """Initialize factor analysis with a factor model.

        Args:
            factor_model: FactorModel instance with factor return data.
        """
        self.factor_model = factor_model
        self.factorCov = self.factor_model.factors.cov()
        self.CV = self.factorCov.values
        self.factors = self.factor_model.factors.columns

    def analyze(self, portfolio: Portfolio) -> FactorResult:
        """Analyze portfolio factor exposures and risk attribution.

        Regresses portfolio returns against factor returns to determine
        factor betas and calculates each factor's contribution to risk.

        Args:
            portfolio: Portfolio instance with time_series and weights (W).

        Returns:
            FactorResult containing betas and risk attribution.
        """
        all_ts = portfolio.time_series.pct_change(1)
        all_ts.dropna(inplace=True)
        port_dates = set([d.strftime("%Y-%m-%d") for d in all_ts.index])
        factor_dates = set(self.factor_model.factors.index)
        common_dates = list(factor_dates.intersection(port_dates))
        all_ts = all_ts.loc[common_dates, :]
        factor_ts = self.factor_model.factors.loc[common_dates, :]

        T = all_ts.values
        P = T @ portfolio.W
        Y = factor_ts.values
        res = sm.OLS(Y, P).fit()

        F = res.params[0]
        S = stdev(P)
        V = S ** 2

        MC = (self.CV @ F.transpose()) / S
        rc = F * MC
        pct_risk = rc / S
        factor_res = FactorResult(
            portfolio, self.factor_model, self.factors, F, S, pct_risk
        )
        return factor_res

