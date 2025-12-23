"""Tests for core module functionality."""

import pytest
from pathlib import Path
import time

from core.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    PORTFOLIOS_FILE,
    HOLDINGS_DIR,
    FACTOR_MODELS_DIR,
    REGIME_DIR,
    get_holdings_file,
    get_factor_model_file,
    get_regime_file
)
from core.timer import timed


class TestPaths:
    """Tests for core.paths module."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should point to existing directory."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_data_dir_exists(self):
        """DATA_DIR should point to models/data directory."""
        assert DATA_DIR.exists()
        assert DATA_DIR.is_dir()
        assert DATA_DIR.name == "data"

    def test_portfolios_file_exists(self):
        """Portfolios CSV file should exist."""
        assert PORTFOLIOS_FILE.exists()
        assert PORTFOLIOS_FILE.suffix == ".csv"

    def test_holdings_dir_exists(self):
        """Holdings directory should exist."""
        assert HOLDINGS_DIR.exists()
        assert HOLDINGS_DIR.is_dir()

    def test_factor_models_dir_exists(self):
        """Factor models directory should exist."""
        assert FACTOR_MODELS_DIR.exists()
        assert FACTOR_MODELS_DIR.is_dir()

    def test_regime_dir_exists(self):
        """Regime directory should exist."""
        assert REGIME_DIR.exists()
        assert REGIME_DIR.is_dir()

    def test_get_holdings_file(self):
        """get_holdings_file should return correct path."""
        path = get_holdings_file(101)
        assert path.name == "port_101.csv"
        assert path.parent == HOLDINGS_DIR

    def test_get_factor_model_file(self):
        """get_factor_model_file should return correct path."""
        path = get_factor_model_file("fama_french")
        assert path.name == "fama_french.csv"
        assert path.parent == FACTOR_MODELS_DIR

    def test_get_regime_file(self):
        """get_regime_file should return correct path."""
        path = get_regime_file("means.csv")
        assert path.name == "means.csv"
        assert path.parent == REGIME_DIR


class TestTimer:
    """Tests for core.timer module."""

    def test_timed_decorator_returns_result(self):
        """Timed decorator should return function result."""
        @timed
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_timed_decorator_preserves_function_name(self):
        """Timed decorator should preserve original function name."""
        @timed
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_timed_decorator_with_kwargs(self):
        """Timed decorator should work with keyword arguments."""
        @timed
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"
