"""Regime-based portfolio analysis for understanding behavior across market conditions.

This module analyzes portfolio performance across different market regimes
(e.g., bull/bear markets, high/low volatility) to understand how the
portfolio behaves in various market conditions.
"""

from models.portfolio import Portfolio
from models.regime_model import RegimeModel
from pandas import DataFrame


class RegimeAnalysis:
    """Analyzes portfolio performance across market regimes.

    Merges portfolio returns with regime classifications to compute
    statistics showing how the portfolio performs in each market regime.

    Attributes:
        regime_model: The RegimeModel with regime definitions.
        all_ts: Holdings time series merged with regime labels.
        proforma: Portfolio P&L time series.
        port_ts: DataFrame of portfolio returns with date index.
        reg_data: Portfolio returns merged with regime classifications.
        reg_stats: Summary statistics of portfolio returns by regime.

    Example:
        regime_model = RegimeModel.load()
        analysis = RegimeAnalysis(portfolio, regime_model)
        print(analysis.reg_stats)  # Mean returns by regime
    """

    def __init__(self, portfolio: Portfolio, regime_model: RegimeModel):
        """Initialize regime analysis for a portfolio.

        Computes portfolio returns, merges with regime data, and calculates
        summary statistics by regime.

        Args:
            portfolio: Portfolio instance with time_series and weights (W).
            regime_model: RegimeModel with regime dates and descriptions.
        """
        self.regime_model = regime_model
        self.all_ts = portfolio.time_series.pct_change(1).dropna()
        self.proforma = self.all_ts.values @ portfolio.W
        self.port_ts = DataFrame(self.proforma, index=self.all_ts.index)
        self.reg_data = self.port_ts.merge(
            self.regime_model.regime_dates, left_index=True, right_on="date"
        )
        self.reg_stats = self.reg_data.groupby("regime").mean()
        self.reg_stats = self.reg_stats.merge(
            regime_model.regime_info, left_on="regime", right_on="regime"
        )
        self.all_ts = self.all_ts.merge(
            regime_model.regime_dates, left_index=True, right_on='date'
        )
        self.stats_by_regime = self.all_ts.groupby("regime").mean()
        print("OK")

