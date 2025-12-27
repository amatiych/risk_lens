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
from core.utils import read_file_from_s3
from regime_analysis_result import RegimeAnalysisReport
import json

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

    regime_info: RegimeAnalysisReport

    @classmethod
    def load(cls,name:str) -> "RegimeModel":
        """Load the regime model from s3.

        Loads regime means, covariances, descriptions, and historical
        regime assignments from the regime data directory.

        Returns:
            RegimeModel instance with all regime data loaded.

        Raises:
            FileNotFoundError: If any regime data files are missing.
        """


        text = read_file_from_s3("risk-lens", f"regimes/{name}.json")
        full_regime_data = RegimeAnalysisReport.from_json(text)

        return RegimeModel(full_regime_data)



if __name__ == "__main__":
    regime = RegimeModel.load("main_regime_model")
    print(regime.regime_info)