"""Market regime model for regime-based risk analysis.

This module provides classes for modeling market regimes and analyzing
portfolio behavior across different market conditions (e.g., bull/bear markets,
high/low volatility periods).
"""

from dataclasses import dataclass
from typing import List
from numpy import zeros
from pandas import DataFrame, read_csv
from datetime import datetime
from core.paths import get_regime_file


class TransitionMatrix:
    """Computes regime transition probabilities from historical data.

    Analyzes a time series of regime classifications to calculate the
    probability of transitioning from one regime to another.

    Attributes:
        txn_probs: NxN matrix of transition probabilities where entry [i,j]
            is the probability of transitioning from regime i to regime j.
    """

    def __init__(self, regime_ts: DataFrame):
        """Initialize transition matrix from regime time series.

        Args:
            regime_ts: DataFrame with 'regime' column containing regime IDs.
        """
        regimes = regime_ts['regime'].values
        regime_list = list(set(regimes))

        n = len(regime_list)
        txn_counts = zeros((n, n), dtype=int)
        for i in range(len(regimes) - 1):
            from_regime_id = regimes[i]
            to_regime_id = regimes[i + 1]
            txn_counts[from_regime_id, to_regime_id] += 1
        row_sums = txn_counts.sum(axis=1, keepdims=True)
        self.txn_probs = txn_counts / row_sums / row_sums


@dataclass
class RegimeModel:
    """Represents a market regime model with historical regime classifications.

    Contains regime-specific statistics including mean returns, covariances,
    and the historical sequence of regime assignments.

    Attributes:
        factors: List of factor names in the model.
        mean_returns: DataFrame of mean returns by regime.
        covariances: DataFrame of covariance matrices by regime.
        regime_dates: DataFrame mapping dates to regime IDs.
        regime_info: DataFrame with regime descriptions and metadata.
    """

    factors: List[str]
    mean_returns: DataFrame
    covariances: DataFrame
    regime_dates: DataFrame
    regime_info: DataFrame

    @classmethod
    def load(cls) -> "RegimeModel":
        """Load the regime model from disk.

        Loads regime means, covariances, descriptions, and historical
        regime assignments from the regime data directory.

        Returns:
            RegimeModel instance with all regime data loaded.

        Raises:
            FileNotFoundError: If any regime data files are missing.
        """
        means = read_csv(get_regime_file("means.csv"))
        covs = read_csv(get_regime_file("covs.csv"))
        regime_info = read_csv(get_regime_file("regime_desc.csv"))
        regime_dates = read_csv(get_regime_file("regimes.csv"))
        regime_dates['date'] = regime_dates['date'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d')
        )
        regime_dates.set_index('date', inplace=True)
        factors = means.columns.values[1:]
        return RegimeModel(factors, means, covs, regime_dates, regime_info)



if __name__ == "__main__":
    regime = RegimeModel.load()
    print(regime.mean_returns)