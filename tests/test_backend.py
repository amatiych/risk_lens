"""Tests for backend module functionality."""

import pytest
import numpy as np
import pandas as pd
from pandas import DataFrame
from unittest.mock import Mock, patch

from backend.risk_engine.var.var_engine import VaR, VarEngine, calc_var_core, calc_expected_shortfall
from backend.risk_engine.factor_analysis import FactorAnalysis, FactorResult
from backend.risk_engine.regime_analysis import RegimeAnalysis
from backend.reporting.portfolio_report import PortfolioReport
from models.portfolio import Portfolio
from models.factor_model import FactorModel
from models.regime_model import RegimeModel


class TestVaR:
    """Tests for VaR results class."""

    def test_var_creation(self):
        """Should create VaR object with all attributes."""
        var = VaR(
            ci=0.95,
            var=0.02,
            k=10,
            var_date=pd.Timestamp('2024-01-15'),
            es=-0.03,
            idx=[1, 2, 3],
            marginal_var=[0.01, 0.005],
            incremental_var=[0.008, 0.004]
        )
        assert var.ci == 0.95
        assert var.var == 0.02
        assert var.es == -0.03
        assert var.var_index == 10
        assert len(var.tail_indexes) == 3

    def test_var_to_json(self):
        """VaR should serialize to JSON."""
        var = VaR(
            ci=0.95,
            var=0.02,
            k=10,
            var_date='2024-01-15',  # Use string instead of Timestamp for JSON
            es=-0.03,
            idx=[1, 2, 3]
        )
        json_str = var.to_json()
        assert isinstance(json_str, str)
        assert '0.95' in json_str


class TestCalcVarCore:
    """Tests for VaR core calculation functions."""

    def test_calc_var_core_single_ci(self):
        """Should calculate VaR for single confidence interval."""
        P = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        cis = np.array([0.90])

        vars_, ks, idxs = calc_var_core(P, cis)

        assert len(vars_) == 1
        assert len(ks) == 1
        assert vars_[0] > 0  # VaR should be positive (loss)

    def test_calc_var_core_multiple_cis(self):
        """Should calculate VaR for multiple confidence intervals."""
        P = np.random.randn(100)
        cis = np.array([0.90, 0.95, 0.99])

        vars_, ks, idxs = calc_var_core(P, cis)

        assert len(vars_) == 3
        assert len(ks) == 3
        # Higher CI should have higher VaR
        assert vars_[2] >= vars_[1] >= vars_[0]

    def test_calc_expected_shortfall(self):
        """Should calculate expected shortfall correctly."""
        P = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        idx = np.argsort(P)  # Sort ascending
        k = 2  # Take worst 2

        es = calc_expected_shortfall(P, idx, k)

        # ES should be average of worst 2 returns
        assert es < 0  # Negative (losses)


class TestVarEngine:
    """Tests for VarEngine class."""

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series for testing."""
        dates = pd.date_range('2024-01-01', periods=100)
        data = {
            'AAPL': np.random.randn(100).cumsum() + 100,
            'MSFT': np.random.randn(100).cumsum() + 200,
            'GOOGL': np.random.randn(100).cumsum() + 150
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def sample_weights(self):
        """Sample portfolio weights."""
        return [0.4, 0.35, 0.25]

    def test_var_engine_creation(self, sample_time_series, sample_weights):
        """Should create VarEngine with time series and weights."""
        engine = VarEngine(sample_time_series, sample_weights)

        assert engine.df_time_series is not None
        assert engine.df_returns is not None
        assert engine.CR is not None  # Correlation matrix

    def test_calc_proforma(self, sample_time_series, sample_weights):
        """Should calculate portfolio P&L."""
        engine = VarEngine(sample_time_series, sample_weights)
        proforma = engine.calc_proforma()

        assert len(proforma) == len(sample_time_series)

    def test_calc_var_returns_list(self, sample_time_series, sample_weights):
        """calc_var should return list of VaR objects."""
        engine = VarEngine(sample_time_series, sample_weights)
        results = engine.calc_var(cis=[0.95])

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], VaR)

    def test_calc_var_multiple_cis(self, sample_time_series, sample_weights):
        """Should calculate VaR for multiple confidence intervals."""
        engine = VarEngine(sample_time_series, sample_weights)
        results = engine.calc_var(cis=[0.95, 0.99])

        assert len(results) == 2
        assert results[0].ci == 0.95
        assert results[1].ci == 0.99

    def test_var_has_marginal_var(self, sample_time_series, sample_weights):
        """VaR results should include marginal VaR."""
        engine = VarEngine(sample_time_series, sample_weights)
        results = engine.calc_var(cis=[0.95])

        assert len(results[0].marginal_var) == len(sample_weights)

    def test_var_has_incremental_var(self, sample_time_series, sample_weights):
        """VaR results should include incremental VaR."""
        engine = VarEngine(sample_time_series, sample_weights)
        results = engine.calc_var(cis=[0.95])

        assert len(results[0].incremental_var) == len(sample_weights)


class TestFactorAnalysis:
    """Tests for FactorAnalysis class."""

    def test_factor_analysis_creation(self):
        """Should create FactorAnalysis with factor model."""
        model = FactorModel.load('fama_french')
        fa = FactorAnalysis(model)

        assert fa.factor_model is not None
        assert fa.factorCov is not None
        assert fa.factors is not None


class TestRegimeAnalysis:
    """Tests for RegimeAnalysis class."""

    @pytest.fixture
    def enriched_portfolio(self):
        """Create a mock enriched portfolio."""
        # Create mock portfolio with time series
        dates = pd.date_range('2020-01-01', periods=300)
        holdings = pd.DataFrame({
            'shares': [100, 50],
        }, index=['AAPL', 'MSFT'])
        holdings.index.name = 'ticker'

        time_series = pd.DataFrame({
            'AAPL': np.random.randn(300).cumsum() + 100,
            'MSFT': np.random.randn(300).cumsum() + 200,
        }, index=dates)

        portfolio = Portfolio(
            portfolio_id=0,
            name="Test",
            nav=10000,
            holdings=holdings,
            W=[0.6, 0.4],
            time_series=time_series
        )
        return portfolio

    def test_regime_analysis_creation(self, enriched_portfolio):
        """Should create RegimeAnalysis with portfolio and regime model."""
        regime_model = RegimeModel.load()
        analysis = RegimeAnalysis(enriched_portfolio, regime_model)

        assert analysis.regime_model is not None
        assert analysis.reg_stats is not None


class TestPortfolioReport:
    """Tests for PortfolioReport class."""

    @pytest.fixture
    def mock_var_results(self):
        """Create mock VaR results."""
        return [VaR(
            ci=0.95,
            var=0.02,
            k=10,
            var_date=pd.Timestamp('2024-01-15'),
            es=-0.03,
            idx=[1, 2, 3],
            marginal_var=[0.01, 0.005],
            incremental_var=[0.008, 0.004]
        )]

    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio."""
        holdings = pd.DataFrame({
            'shares': [100, 50],
            'price': [150.0, 300.0],
            'market_value': [15000.0, 15000.0],
            'weight': [0.5, 0.5]
        }, index=['AAPL', 'MSFT'])
        holdings.index.name = 'ticker'

        return Portfolio(
            portfolio_id=0,
            name="Test",
            nav=30000,
            holdings=holdings
        )

    @pytest.fixture
    def mock_correlation(self):
        """Create mock correlation matrix."""
        return pd.DataFrame({
            'AAPL': [1.0, 0.7],
            'MSFT': [0.7, 1.0]
        }, index=['AAPL', 'MSFT'])

    @pytest.fixture
    def mock_factor_result(self):
        """Create mock factor result."""
        mock = Mock(spec=FactorResult)
        mock.factors = ['Mkt-RF', 'SMB', 'HML']
        mock.betas = [1.1, 0.3, -0.2]
        mock.marginal_risk = [0.8, 0.1, 0.05]
        mock.factor_model = Mock()
        mock.factor_model.model_name = 'fama_french'
        return mock

    @pytest.fixture
    def mock_regime_analysis(self):
        """Create mock regime analysis."""
        mock = Mock()
        mock.reg_stats = pd.DataFrame({
            'regime': [0, 1],
            'mean_return': [0.001, -0.002]
        })
        mock.all_ts = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'return': np.random.randn(10) * 0.01
        })
        mock.stats_by_regime = pd.DataFrame({
            'regime': [0, 1],
            'mean': [0.001, -0.002]
        })
        return mock

    def test_portfolio_report_creation(
        self,
        mock_portfolio,
        mock_var_results,
        mock_correlation,
        mock_factor_result,
        mock_regime_analysis
    ):
        """Should create PortfolioReport with all components."""
        report = PortfolioReport(
            mock_portfolio,
            mock_var_results,
            mock_correlation,
            mock_factor_result,
            mock_regime_analysis
        )

        assert report.portfolio is not None
        assert report.var is not None
        assert report.report is not None

    def test_portfolio_report_has_holdings_json(
        self,
        mock_portfolio,
        mock_var_results,
        mock_correlation,
        mock_factor_result,
        mock_regime_analysis
    ):
        """Report should have holdings JSON."""
        report = PortfolioReport(
            mock_portfolio,
            mock_var_results,
            mock_correlation,
            mock_factor_result,
            mock_regime_analysis
        )

        assert report.holdings_json is not None
        assert isinstance(report.holdings_json, str)

    def test_portfolio_report_has_var_json(
        self,
        mock_portfolio,
        mock_var_results,
        mock_correlation,
        mock_factor_result,
        mock_regime_analysis
    ):
        """Report should have VaR JSON."""
        report = PortfolioReport(
            mock_portfolio,
            mock_var_results,
            mock_correlation,
            mock_factor_result,
            mock_regime_analysis
        )

        assert report.var_json is not None
        assert isinstance(report.var_json, str)
