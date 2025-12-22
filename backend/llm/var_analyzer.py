import os
from datetime import datetime
import anthropic
import json
from backend.reporting.portfolio_report import PortfolioReport


class VaRAnalyzer:

    def __init__(self,report:PortfolioReport):
        self.api_key = os.environ.get("CLAUDE_API_KEY")
        self.report = report

    def analyze(self):
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
        claude = anthropic.Anthropic(api_key=self.api_key)
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Clean JSON
        clean_json = response_text
        if '```json' in clean_json:
            clean_json = clean_json.split('```json')[1].split('```')[0]
        elif '```' in clean_json:
            clean_json = clean_json.split('```')[1].split('```')[0]

        clean_json = clean_json.strip()

        return clean_json