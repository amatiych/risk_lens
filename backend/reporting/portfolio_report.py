"""Portfolio report generation for comprehensive risk reporting.

This module provides the PortfolioReport class which consolidates all
risk analysis results into a structured report format suitable for
display and AI analysis.
"""

from dataclasses import dataclass
from typing import List, Optional

from backend.risk_engine.factor_analysis import FactorResult
from backend.risk_engine.regime_analysis import RegimeAnalysis
from backend.risk_engine.var.var_engine import VaR
from models.portfolio import Portfolio
import json
from pandas import DataFrame


class PortfolioReport:
    """Comprehensive portfolio risk report consolidating all analysis results.

    Aggregates VaR, factor, and regime analysis into a structured report
    with JSON-serialized data for display and AI consumption.

    Attributes:
        portfolio: The analyzed Portfolio instance.
        var: List of VaR results at different confidence levels.
        cr: Correlation matrix DataFrame.
        factor_res: Factor analysis results.
        holdings_json: JSON string of holdings data.
        var_json: JSON string of VaR results.
        mvar_json: JSON string of marginal/incremental VaR.
        factor_data: Dictionary of factor betas and risk contributions.
        report: Formatted text report combining all analysis.

    Example:
        report = PortfolioReport(portfolio, var_results, corr, factors, regimes)
        print(report.report)  # Full text report
    """

    def __init__(
        self,
        portfolio: Portfolio,
        var: List[VaR],
        cr: Optional[DataFrame] = None,
        factor_res: Optional[FactorResult] = None,
        regime_anlaysis: Optional[RegimeAnalysis] = None
    ):
        """Initialize portfolio report with all analysis components.

        Args:
            portfolio: Portfolio instance with holdings and weights.
            var: List of VaR results from VarEngine.
            cr: Correlation matrix of holdings returns.
            factor_res: FactorResult from factor analysis.
            regime_anlaysis: RegimeAnalysis results.
        """
        self.portfolio = portfolio
        self.var = var
        self.cr = cr
        self.factor_res = factor_res
        self.holdings_json = self.portfolio.holdings.reset_index().to_json(
            orient='records'
        )

        mvs = self.portfolio.holdings.market_value.values
        net_value = sum(mvs)
        gross_value = sum([abs(w) for w in mvs])
        var_results = []
        marginal_var = []

        tickers = self.portfolio.holdings.index.tolist()
        for var_item in self.var:
            var_results.append({
                'CI': var_item.ci,
                'Date': var_item.var_date.strftime("%Y-%m-%d"),
                'VaR': var_item.var,
                'ES': var_item.es
            })
            for ticker, mv, iv in zip(
                tickers, var_item.marginal_var, var_item.incremental_var
            ):
                marginal_var.append({
                    'CI': var_item.ci,
                    'Ticker': ticker,
                    'Marginal VaR': mv,
                    'Incremental VaR': iv
                })
        self.var_json = json.dumps(var_results)
        self.mvar_json = json.dumps(marginal_var)

        self.factor_data = {
            factor: {'beta': float(beta), '% risk': float(mr)}
            for factor, beta, mr in zip(
                self.factor_res.factors,
                self.factor_res.betas,
                self.factor_res.marginal_risk
            )
        }

        self.report = f"""
        Holdings:
        {self.holdings_json}

        NAV: {portfolio.nav}  Net Market Value: {net_value} Gross Market Value: {gross_value}
        NAV:  Net Exposure : {net_value/portfolio.nav} Gross Exposure: {gross_value/portfolio.nav}

        Value at Risk Report
        {self.var_json}

        Marginal and Incremental VaR Report:
        {self.mvar_json}

        Correlation Matrix
        {self.cr.reset_index().to_json(orient='records', double_precision=2)}

        Factor Analysis
        Factor Model: {self.factor_res.factor_model.model_name}
        Factors Betas and Risk Contribution:     {self.factor_data}

        Regime Analysis:
        Portfolio Regime Stats: {regime_anlaysis.reg_stats.to_json(orient='records', double_precision=2)}

        Holdings Time Series with Regime Data:
        {regime_anlaysis.stats_by_regime.to_json(orient='records', double_precision=2)}
        """



