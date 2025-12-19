import os

import anthropic
import json
from backend.reporting.portfolio_report import PortfolioReport


class VaRAnalyzer:

    def __init__(self,report:PortfolioReport):
        self.api_key = os.environ.get("CLAUDE_API_KEY")
        self.report = report

    def analyze(self):
        prompt = f"""
            You are a senior risk manager for a hedge fund. Review The results of the VaR Analysis shown below:
            
            1. Identify Largest drivers of risk in terms of marginal and incremental VaR. 
            2. Identify Largest concentration and best diversifiers within the portfolio. 
            3. Use Yahoo Finance. Look at the date for the VaR and identify any risk factors that may have caused the loss that day
            4. Enrich portfolio information with industry / sector information for each position. 
            
            Use only factual data. Do not make up facts. If unsure return no results. 
            
            VaR Report : {self.report.report}
            
            
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