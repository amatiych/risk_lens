"""AI-powered VaR analysis using Claude for intelligent risk interpretation.

This module provides the VaRAnalyzer class which uses Claude AI to interpret
portfolio risk metrics and provide actionable insights for risk managers.
"""

import os
from datetime import datetime
import anthropic
import json
from backend.reporting.portfolio_report import PortfolioReport


class VaRAnalyzer:
    """Uses Claude AI to analyze and interpret VaR reports.

    Takes a portfolio risk report and generates comprehensive analysis
    including risk drivers, concentration analysis, factor interpretation,
    and regime-based insights.

    Attributes:
        api_key: Anthropic API key from CLAUDE_API_KEY environment variable.
        report: PortfolioReport instance containing risk metrics.

    Example:
        report = PortfolioReport(portfolio, var_results, correlation, factors, regimes)
        analyzer = VaRAnalyzer(report)
        analysis = analyzer.analyze()
        print(analysis)
    """

    def __init__(self, report: PortfolioReport):
        """Initialize the VaR analyzer with a portfolio report.

        Args:
            report: PortfolioReport containing VaR, factor, and regime data.
        """
        self.api_key = os.environ.get("CLAUDE_API_KEY")
        self.report = report

    def analyze(self) -> str:
        """Generate AI-powered analysis of the portfolio risk report.

        Sends the risk report to Claude for interpretation, requesting:
        - Identification of largest risk drivers
        - Concentration and diversification analysis
        - Historical context for VaR events
        - Sector/industry enrichment
        - Factor exposure interpretation
        - Regime analysis interpretation

        Returns:
            String containing Claude's analysis of the portfolio risk.

        Raises:
            anthropic.APIError: If the API call fails.
        """
        prompt = f"""
            You are a senior risk manager for a hedge fund.
            Review The results of the VaR Analysis shown below:

            Today is {datetime.today().strftime('%Y-%m-%d')}

            1. Identify Largest drivers of risk in terms of marginal and incremental VaR.
            2. Identify Largest concentration and best diversifiers within the portfolio.
            3. Use Yahoo Finance. Look at the date for the VaR and identify any risk factors that may have caused the loss that day
            4. Use Yahoo Finance and enrich portfolio information with industry / sector information for each position.
            5. Interpret Factor Exposures and Contribution to Portfolio Volatility.
            6. Interpret Regime Analysis results for the portfolio
            7. Review regime information for holdings time series
            Use only factual data. Do not make up facts. If unsure return no results.

            Risk Report : {self.report.report}


        """
        open("/tmp/claude_prompt.txt","w").write(prompt)
        claude = anthropic.Anthropic(api_key=self.api_key)
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Clean JSON if response is wrapped in code blocks
        clean_json = response_text
        if '```json' in clean_json:
            clean_json = clean_json.split('```json')[1].split('```')[0]
        elif '```' in clean_json:
            clean_json = clean_json.split('```')[1].split('```')[0]

        clean_json = clean_json.strip()

        return clean_json