"""Centralized path configuration for the Risk Lens application.

This module provides absolute paths to all data files and directories,
ensuring the application works correctly regardless of the current
working directory.

Attributes:
    PROJECT_ROOT: Absolute path to the project root directory.
    DATA_DIR: Path to the main data directory (models/data).
    PORTFOLIOS_FILE: Path to the portfolios metadata CSV.
    HOLDINGS_DIR: Directory containing portfolio holdings files.
    FACTOR_MODELS_DIR: Directory containing factor model data.
    REGIME_DIR: Directory containing market regime data.
    TIME_SERIES_FILE: Path to the cached time series data.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJECT_ROOT / "models" / "data"

PORTFOLIOS_FILE = DATA_DIR / "portfolios.csv"
HOLDINGS_DIR = DATA_DIR / "holdings"
FACTOR_MODELS_DIR = DATA_DIR / "factor_models"
REGIME_DIR = DATA_DIR / "regime"
TIME_SERIES_FILE = DATA_DIR / "time_series.csv"


def get_holdings_file(portfolio_id: int) -> Path:
    """Get the path to a portfolio's holdings file.

    Args:
        portfolio_id: The unique identifier for the portfolio.

    Returns:
        Path to the holdings CSV file (e.g., holdings/port_101.csv).
    """
    return HOLDINGS_DIR / f"port_{portfolio_id}.csv"


def get_factor_model_file(model_name: str) -> Path:
    """Get the path to a factor model file.

    Args:
        model_name: Name of the factor model (e.g., 'fama_french').

    Returns:
        Path to the factor model CSV file.
    """
    return FACTOR_MODELS_DIR / f"{model_name}.csv"


def get_regime_file(filename: str) -> Path:
    """Get the path to a regime data file.

    Args:
        filename: Name of the regime file (e.g., 'means.csv', 'regimes.csv').

    Returns:
        Path to the regime data file.
    """
    return REGIME_DIR / filename
