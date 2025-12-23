"""Tests for frontend service functionality."""

import pytest
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import Mock, patch

from frontend.services.portfolio_service import (
    parse_uploaded_csv,
    AnalysisResults
)
from frontend.utils.formatters import (
    format_currency,
    format_percentage,
    format_var_table,
    format_holdings_table,
    format_factor_table
)
from backend.risk_engine.var.var_engine import VaR


class TestParseUploadedCsv:
    """Tests for parse_uploaded_csv function."""

    def test_parse_valid_csv(self):
        """Should parse valid CSV with ticker and shares."""
        csv_content = "ticker,shares\nAAPL,100\nMSFT,50\nGOOGL,25"
        uploaded_file = StringIO(csv_content)

        portfolio = parse_uploaded_csv(uploaded_file)

        assert portfolio is not None
        assert len(portfolio.holdings) == 3
        assert 'AAPL' in portfolio.holdings.index
        assert portfolio.holdings.loc['AAPL', 'shares'] == 100

    def test_parse_csv_with_nav(self):
        """Should parse CSV with custom NAV."""
        csv_content = "ticker,shares\nAAPL,100"
        uploaded_file = StringIO(csv_content)

        portfolio = parse_uploaded_csv(uploaded_file, nav=50000.0)

        assert portfolio.nav == 50000.0

    def test_parse_csv_default_nav(self):
        """Should use default NAV of 1.0 when not specified."""
        csv_content = "ticker,shares\nAAPL,100"
        uploaded_file = StringIO(csv_content)

        portfolio = parse_uploaded_csv(uploaded_file)

        assert portfolio.nav == 1.0

    def test_parse_csv_uppercase_tickers(self):
        """Should convert tickers to uppercase."""
        csv_content = "ticker,shares\naapl,100\nmsft,50"
        uploaded_file = StringIO(csv_content)

        portfolio = parse_uploaded_csv(uploaded_file)

        assert 'AAPL' in portfolio.holdings.index
        assert 'MSFT' in portfolio.holdings.index

    def test_parse_csv_strips_whitespace(self):
        """Should strip whitespace from tickers."""
        csv_content = "ticker,shares\n  AAPL  ,100"
        uploaded_file = StringIO(csv_content)

        portfolio = parse_uploaded_csv(uploaded_file)

        assert 'AAPL' in portfolio.holdings.index

    def test_parse_csv_missing_ticker_column(self):
        """Should raise error if ticker column missing."""
        csv_content = "symbol,shares\nAAPL,100"
        uploaded_file = StringIO(csv_content)

        with pytest.raises(ValueError) as excinfo:
            parse_uploaded_csv(uploaded_file)

        assert "ticker" in str(excinfo.value).lower()

    def test_parse_csv_missing_shares_column(self):
        """Should raise error if shares column missing."""
        csv_content = "ticker,quantity\nAAPL,100"
        uploaded_file = StringIO(csv_content)

        with pytest.raises(ValueError) as excinfo:
            parse_uploaded_csv(uploaded_file)

        assert "shares" in str(excinfo.value).lower()


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_format_millions(self):
        """Should format large values in millions."""
        result = format_currency(1500000)
        assert 'M' in result
        assert '$' in result

    def test_format_thousands(self):
        """Should format values in thousands."""
        result = format_currency(5000)
        assert 'K' in result
        assert '$' in result

    def test_format_small_values(self):
        """Should format small values normally."""
        result = format_currency(500)
        assert '$' in result
        assert '500' in result

    def test_format_negative_values(self):
        """Should handle negative values."""
        result = format_currency(-1000000)
        assert 'M' in result


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_format_decimal_to_percent(self):
        """Should convert decimal to percentage."""
        result = format_percentage(0.15)
        assert '15' in result
        assert '%' in result

    def test_format_with_decimals(self):
        """Should format with specified decimal places."""
        result = format_percentage(0.12345, decimals=2)
        assert '12.35%' in result or '12.34%' in result  # Rounding

    def test_format_negative_percentage(self):
        """Should handle negative percentages."""
        result = format_percentage(-0.05)
        assert '-' in result
        assert '%' in result


class TestFormatVarTable:
    """Tests for format_var_table function."""

    def test_format_var_table_structure(self):
        """Should return DataFrame with correct columns."""
        var_results = [VaR(
            ci=0.95,
            var=0.02,
            k=10,
            var_date=pd.Timestamp('2024-01-15'),
            es=-0.03,
            idx=[1, 2, 3]
        )]

        df = format_var_table(var_results)

        assert isinstance(df, pd.DataFrame)
        assert 'Confidence Level' in df.columns
        assert 'VaR' in df.columns
        assert 'Expected Shortfall' in df.columns

    def test_format_var_table_multiple_cis(self):
        """Should handle multiple confidence intervals."""
        var_results = [
            VaR(ci=0.95, var=0.02, k=10, var_date=pd.Timestamp('2024-01-15'), es=-0.03, idx=[]),
            VaR(ci=0.99, var=0.04, k=5, var_date=pd.Timestamp('2024-01-10'), es=-0.05, idx=[])
        ]

        df = format_var_table(var_results)

        assert len(df) == 2


class TestFormatHoldingsTable:
    """Tests for format_holdings_table function."""

    def test_format_holdings_structure(self):
        """Should return DataFrame with correct columns."""
        holdings = pd.DataFrame({
            'shares': [100, 50],
            'price': [150.0, 300.0],
            'market_value': [15000.0, 15000.0],
            'weight': [0.5, 0.5]
        }, index=['AAPL', 'MSFT'])
        holdings.index.name = 'ticker'

        df = format_holdings_table(holdings)

        assert 'Ticker' in df.columns
        assert 'Shares' in df.columns
        assert 'Price' in df.columns
        assert 'Market Value' in df.columns
        assert 'Weight' in df.columns


class TestFormatFactorTable:
    """Tests for format_factor_table function."""

    def test_format_factor_table_structure(self):
        """Should return DataFrame with correct columns."""
        mock_result = Mock()
        mock_result.factors = ['Mkt-RF', 'SMB', 'HML']
        mock_result.betas = [1.1, 0.3, -0.2]
        mock_result.marginal_risk = [0.8, 0.1, 0.05]

        df = format_factor_table(mock_result)

        assert isinstance(df, pd.DataFrame)
        assert 'Factor' in df.columns
        assert 'Beta' in df.columns
        assert 'Risk Contribution' in df.columns
        assert len(df) == 3


class TestAnalysisResults:
    """Tests for AnalysisResults dataclass."""

    def test_analysis_results_creation(self):
        """Should create AnalysisResults with all attributes."""
        mock_portfolio = Mock()
        mock_var_results = []
        mock_correlation = pd.DataFrame()
        mock_factor_result = Mock()
        mock_regime_analysis = Mock()
        mock_report = Mock()

        results = AnalysisResults(
            portfolio=mock_portfolio,
            var_results=mock_var_results,
            correlation_matrix=mock_correlation,
            factor_result=mock_factor_result,
            regime_analysis=mock_regime_analysis,
            report=mock_report,
            tickers=['AAPL', 'MSFT']
        )

        assert results.portfolio is mock_portfolio
        assert results.tickers == ['AAPL', 'MSFT']
