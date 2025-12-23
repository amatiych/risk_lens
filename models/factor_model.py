"""Factor model data structures for multi-factor risk analysis.

This module provides the FactorModel class for loading and representing
factor models such as Fama-French for portfolio factor exposure analysis.
"""

from dataclasses import dataclass
from pandas import DataFrame, read_csv
from core.paths import get_factor_model_file


@dataclass
class FactorModel:
    """Represents a multi-factor model for risk analysis.

    A factor model contains historical factor returns used to decompose
    portfolio returns into systematic factor exposures.

    Attributes:
        model_name: Name identifier for the model (e.g., 'fama_french').
        factors: DataFrame with Date index and factor return columns.

    Example:
        model = FactorModel.load('fama_french')
        # Access factors like: model.factors['Mkt-RF'], model.factors['SMB']
    """

    model_name: str
    factors: DataFrame

    @classmethod
    def load(cls, model_name: str) -> 'FactorModel':
        """Load a factor model from disk.

        Args:
            model_name: Name of the factor model to load (e.g., 'fama_french').

        Returns:
            FactorModel instance with factor returns DataFrame.

        Raises:
            FileNotFoundError: If the factor model file does not exist.
        """
        factors = read_csv(get_factor_model_file(model_name)).set_index('Date')
        return FactorModel(model_name=model_name, factors=factors)

if __name__ == '__main__':
    factor_model = FactorModel.load('fama_french')
    print(factor_model.factors)