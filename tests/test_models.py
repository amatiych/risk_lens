"""Tests for models module functionality."""

import pytest
import pandas as pd
from pandas import DataFrame

from models.portfolio import Portfolio
from models.factor_model import FactorModel
from models.regime_model import RegimeModel, TransitionMatrix


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_load_portfolio_by_id(self):
        """Should load a portfolio by ID."""
        portfolio = Portfolio.load(101)
        assert portfolio is not None
        assert portfolio.portfolio_id == 101
        assert isinstance(portfolio.holdings, DataFrame)

    def test_portfolio_has_required_attributes(self):
        """Portfolio should have all required attributes."""
        portfolio = Portfolio.load(101)
        assert hasattr(portfolio, 'portfolio_id')
        assert hasattr(portfolio, 'name')
        assert hasattr(portfolio, 'nav')
        assert hasattr(portfolio, 'holdings')
        assert hasattr(portfolio, 'W')
        assert hasattr(portfolio, 'time_series')

    def test_get_portfolios_returns_dataframe(self):
        """get_portfolios should return a DataFrame."""
        portfolios = Portfolio.get_portfolios()
        assert isinstance(portfolios, DataFrame)
        assert 'name' in portfolios.columns
        assert 'nav' in portfolios.columns

    def test_holdings_has_ticker_index(self):
        """Holdings DataFrame should have ticker as index."""
        portfolio = Portfolio.load(101)
        assert portfolio.holdings.index.name == 'ticker'

    def test_portfolio_nav_is_numeric(self):
        """Portfolio NAV should be a numeric value."""
        import numpy as np
        portfolio = Portfolio.load(101)
        assert isinstance(portfolio.nav, (int, float, np.integer, np.floating))
        assert portfolio.nav > 0


class TestFactorModel:
    """Tests for FactorModel class."""

    def test_load_fama_french_model(self):
        """Should load the Fama-French factor model."""
        model = FactorModel.load('fama_french')
        assert model is not None
        assert model.model_name == 'fama_french'

    def test_factor_model_has_factors_dataframe(self):
        """Factor model should have factors DataFrame."""
        model = FactorModel.load('fama_french')
        assert isinstance(model.factors, DataFrame)
        assert len(model.factors) > 0

    def test_factors_have_date_index(self):
        """Factors DataFrame should have Date index."""
        model = FactorModel.load('fama_french')
        assert model.factors.index.name == 'Date'

    def test_factors_have_columns(self):
        """Factors DataFrame should have factor columns."""
        model = FactorModel.load('fama_french')
        assert len(model.factors.columns) > 0


class TestRegimeModel:
    """Tests for RegimeModel class."""

    def test_load_regime_model(self):
        """Should load the regime model."""
        model = RegimeModel.load()
        assert model is not None

    def test_regime_model_has_required_attributes(self):
        """Regime model should have all required attributes."""
        model = RegimeModel.load()
        assert hasattr(model, 'factors')
        assert hasattr(model, 'mean_returns')
        assert hasattr(model, 'covariances')
        assert hasattr(model, 'regime_dates')
        assert hasattr(model, 'regime_info')

    def test_regime_dates_is_dataframe(self):
        """regime_dates should be a DataFrame."""
        model = RegimeModel.load()
        assert isinstance(model.regime_dates, DataFrame)

    def test_regime_info_is_dataframe(self):
        """regime_info should be a DataFrame."""
        model = RegimeModel.load()
        assert isinstance(model.regime_info, DataFrame)

    def test_mean_returns_is_dataframe(self):
        """mean_returns should be a DataFrame."""
        model = RegimeModel.load()
        assert isinstance(model.mean_returns, DataFrame)


class TestTransitionMatrix:
    """Tests for TransitionMatrix class."""

    def test_transition_matrix_creation(self):
        """Should create transition matrix from regime data."""
        model = RegimeModel.load()
        txn_matrix = TransitionMatrix(model.regime_dates.reset_index())
        assert hasattr(txn_matrix, 'txn_probs')

    def test_transition_probs_shape(self):
        """Transition probabilities should be square matrix."""
        model = RegimeModel.load()
        txn_matrix = TransitionMatrix(model.regime_dates.reset_index())
        n_rows, n_cols = txn_matrix.txn_probs.shape
        assert n_rows == n_cols
