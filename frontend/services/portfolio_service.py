"""Portfolio processing and analysis orchestration service.

This module provides functions for parsing uploaded portfolio files,
enriching them with market data, and running comprehensive risk analysis.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from models.portfolio import Portfolio
from models.enrichment.price_enricher import YahooFinancePriceEnricher
from models.enrichment.time_series_enricher import YahooTimeSeriesEnricher
from models.factor_model import FactorModel
from models.regime_model import RegimeModel
from backend.risk_engine.var.var_engine import VarEngine, VaR
from backend.risk_engine.factor_analysis import FactorAnalysis, FactorResult
from backend.risk_engine.regime_analysis import RegimeAnalysis
from backend.reporting.portfolio_report import PortfolioReport


@dataclass
class AnalysisResults:
    """Container for all portfolio analysis results.

    Attributes:
        portfolio: The analyzed Portfolio instance.
        var_results: List of VaR calculations at different confidence levels.
        correlation_matrix: DataFrame of asset return correlations.
        factor_result: Factor analysis results.
        regime_analysis: Regime-based analysis results.
        report: Consolidated PortfolioReport.
        tickers: List of ticker symbols in the portfolio.
    """

    portfolio: Portfolio
    var_results: List[VaR]
    correlation_matrix: pd.DataFrame
    factor_result: FactorResult
    regime_analysis: RegimeAnalysis
    report: PortfolioReport
    tickers: List[str]


def parse_uploaded_csv(uploaded_file, nav: Optional[float] = None) -> Portfolio:
    """Parse uploaded CSV file into a Portfolio object.

    Args:
        uploaded_file: Streamlit UploadedFile object containing CSV data.
        nav: Optional Net Asset Value. If None, calculated from market values.

    Returns:
        Portfolio instance with holdings from the CSV.

    Raises:
        ValueError: If required columns (ticker, shares) are missing.
    """
    df = pd.read_csv(uploaded_file)

    required_cols = ['ticker', 'shares']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[['ticker', 'shares']].copy()
    df['ticker'] = df['ticker'].str.upper().str.strip()
    df = df.set_index('ticker')

    portfolio = Portfolio(
        portfolio_id=0,
        name="Uploaded Portfolio",
        nav=nav if nav else 1.0,
        holdings=df
    )

    return portfolio


def enrich_portfolio(portfolio: Portfolio) -> Portfolio:
    """Enrich portfolio with current prices and historical time series.

    Fetches current market prices from Yahoo Finance to calculate
    market values and weights, then fetches 12 months of historical
    prices for risk calculations.

    Args:
        portfolio: Portfolio instance to enrich.

    Returns:
        The same portfolio instance with price, market_value, weight,
        and time_series data populated.
    """
    enrichers = [YahooFinancePriceEnricher(), YahooTimeSeriesEnricher()]
    for enricher in enrichers:
        enricher.enrich_portfolio(portfolio)
    return portfolio


def run_analysis(
    portfolio: Portfolio,
    confidence_levels: List[float] = [0.95, 0.99]
) -> AnalysisResults:
    """Run comprehensive risk analysis on an enriched portfolio.

    Executes VaR calculation, factor analysis, and regime analysis,
    then consolidates results into an AnalysisResults container.

    Args:
        portfolio: Enriched Portfolio with time_series and weights.
        confidence_levels: VaR confidence intervals (default [0.95, 0.99]).

    Returns:
        AnalysisResults containing all analysis outputs.
    """
    tickers = list(portfolio.time_series.columns)
    weights = list(portfolio.holdings.loc[tickers, 'weight'].values)

    var_engine = VarEngine(portfolio.time_series, weights)
    var_results = var_engine.calc_var(cis=confidence_levels)
    correlation_matrix = var_engine.CR

    factor_model = FactorModel.load('fama_french')
    fa = FactorAnalysis(factor_model)
    factor_result = fa.analyze(portfolio)

    regime_model = RegimeModel.load("main_regime_model")
    regime_analysis = RegimeAnalysis(portfolio, regime_model)

    report = PortfolioReport(
        portfolio,
        var_results,
        correlation_matrix,
        factor_result,
        regime_analysis
    )

    return AnalysisResults(
        portfolio=portfolio,
        var_results=var_results,
        correlation_matrix=correlation_matrix,
        factor_result=factor_result,
        regime_analysis=regime_analysis,
        report=report,
        tickers=tickers
    )
