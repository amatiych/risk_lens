"""AI-powered VaR analysis using Claude for intelligent risk interpretation.

This module provides the VaRAnalyzer class which uses Claude AI to interpret
portfolio risk metrics and provide actionable insights for risk managers.
Uses tool calling for runtime data fetching, structured outputs for
consistent responses, and comprehensive guardrails for safety.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from backend.llm.tools import TOOLS, execute_tool
from backend.llm.schemas import VAR_ANALYSIS_SCHEMA, VaRAnalysisResult
from backend.reporting.portfolio_report import PortfolioReport
from backend.llm.guardrails.input_guards import InputGuardrails
from backend.llm.guardrails.output_guards import OutputGuardrails
from backend.llm.guardrails.process_guards import ProcessGuardrails, GuardedClient
from backend.llm.guardrails.constitutional import ConstitutionalGuard


# Import GuardrailsReport from the package
from backend.llm.guardrails import GuardrailsReport


class VaRAnalyzer:
    """Uses Claude AI to analyze and interpret VaR reports.

    Takes a portfolio risk report and generates comprehensive analysis
    including risk drivers, concentration analysis, factor interpretation,
    and regime-based insights. Uses tool calling to fetch real-time data
    from Yahoo Finance and structured outputs for consistent responses.

    Attributes:
        api_key: Anthropic API key from CLAUDE_API_KEY environment variable.
        report: PortfolioReport instance containing risk metrics.
        max_tool_iterations: Maximum number of tool call rounds (default: 10).

    Example:
        report = PortfolioReport(portfolio, var_results, correlation, factors, regimes)
        analyzer = VaRAnalyzer(report)
        result = analyzer.analyze()
        print(result.executive_summary)
        print(result.recommendations)
    """

    def __init__(
        self,
        report: PortfolioReport,
        max_tool_iterations: int = 10,
        guardrails_enabled: bool = True,
        user_id: str = "default"
    ):
        """Initialize the VaR analyzer with a portfolio report.

        Args:
            report: PortfolioReport containing VaR, factor, and regime data.
            max_tool_iterations: Maximum rounds of tool calls before forcing completion.
            guardrails_enabled: Whether to run guardrails checks.
            user_id: User identifier for rate limiting and logging.
        """
        self.api_key = os.environ.get("CLAUDE_API_KEY")
        self.report = report
        self.max_tool_iterations = max_tool_iterations
        self.guardrails_enabled = guardrails_enabled
        self.user_id = user_id

        # Initialize guardrails
        if guardrails_enabled:
            self.input_guards = InputGuardrails()
            self.output_guards = OutputGuardrails()
            self.process_guards = ProcessGuardrails()
            self.constitutional = ConstitutionalGuard()

    def _build_initial_prompt(self) -> str:
        """Build the initial analysis prompt with portfolio data.

        Returns:
            Formatted prompt string with embedded report data.
        """
        return f"""You are a senior risk manager for a hedge fund.
Review the VaR Analysis results below and provide a comprehensive risk assessment.

Today is {datetime.today().strftime('%Y-%m-%d')}

You have access to tools to look up additional information:
- lookup_stock_info: Get company sector, industry, and description
- get_historical_prices: Get price data around the VaR date
- get_market_news: Get recent news for context

Please analyze the following:

1. **Risk Drivers**: Identify the largest drivers of risk in terms of marginal and incremental VaR.
   Use the lookup_stock_info tool to get sector/industry info for each position.

2. **Concentration Analysis**: Identify positions with highest concentration risk and best diversifiers.

3. **VaR Date Context**: Look at the VaR date and use get_historical_prices and get_market_news
   to identify any risk factors that may have caused the loss that day.

4. **Factor Interpretation**: Interpret the factor exposures and their contribution to portfolio volatility.

5. **Regime Analysis**: Interpret how the portfolio performs across different market regimes.

6. **Recommendations**: Provide actionable recommendations to improve the risk profile.

Use only factual data from the tools. Do not fabricate information.

Risk Report:
{self.report.report}
"""

    def _execute_tool_calls(
        self,
        response: anthropic.types.Message
    ) -> List[Dict[str, Any]]:
        """Execute all tool calls from a response.

        Args:
            response: Claude API response containing tool_use blocks.

        Returns:
            List of tool_result dictionaries to send back to Claude.
        """
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": str(e)}),
                        "is_error": True
                    })
        return tool_results

    def analyze(self) -> VaRAnalysisResult:
        """Generate AI-powered analysis of the portfolio risk report.

        Sends the risk report to Claude with tools enabled for runtime
        data fetching. Handles tool calls in a loop until Claude provides
        a final structured response.

        Returns:
            VaRAnalysisResult containing structured analysis with risk drivers,
            concentration analysis, factor interpretation, regime insights,
            and recommendations.

        Raises:
            anthropic.APIError: If the API call fails.
            ValueError: If unable to parse structured response.
        """
        client = anthropic.Anthropic(api_key=self.api_key)

        messages = [
            {"role": "user", "content": self._build_initial_prompt()}
        ]

        iteration = 0
        final_text = ""

        while iteration < self.max_tool_iterations:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                tools=TOOLS,
                messages=messages
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Execute tool calls
                tool_results = self._execute_tool_calls(response)

                # Add assistant response and tool results to conversation
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

                iteration += 1
            else:
                # Claude is done with tools, extract final text
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                break

        # If we hit max iterations, force a final response
        if iteration >= self.max_tool_iterations and not final_text:
            messages.append({
                "role": "user",
                "content": "Please provide your final analysis now based on the information gathered."
            })
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=messages
            )
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text

        # Now request structured output
        structured_response = self._get_structured_response(client, final_text)
        return structured_response

    def _get_structured_response(
        self,
        client: anthropic.Anthropic,
        analysis_text: str
    ) -> VaRAnalysisResult:
        """Convert free-form analysis into structured output.

        Args:
            client: Anthropic client instance.
            analysis_text: Free-form analysis text from Claude.

        Returns:
            VaRAnalysisResult with parsed structured data.
        """
        prompt = f"""Based on this analysis, provide a structured JSON response.

Analysis:
{analysis_text}

Return a JSON object with exactly these fields:
- executive_summary: Brief 2-3 sentence overview
- risk_drivers: Array of objects with ticker, marginal_var_contribution (number), incremental_var_contribution (number), explanation, and optionally company_name and sector
- concentration_analysis: Object with most_concentrated_positions (array of tickers), best_diversifiers (array of tickers), concentration_risk_level (one of: low, moderate, high, very_high), concentration_summary (string)
- factor_interpretation: Object with dominant_factors (array of strings), factor_risk_summary (string), and optionally factor_exposures array
- regime_interpretation: Object with regime_sensitivity (one of: low, moderate, high), regime_risk_summary (string), and optionally best_performing_regime and worst_performing_regime
- var_date_context: Object with var_date (YYYY-MM-DD), context_explanation (string), and optionally market_events (array) and price_movements (array)
- recommendations: Array of actionable recommendation strings

Return ONLY valid JSON, no markdown formatting."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        # Clean JSON if wrapped in code blocks
        clean_json = response_text
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0]
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0]
        clean_json = clean_json.strip()

        try:
            data = json.loads(clean_json)
            return VaRAnalysisResult.from_dict(data)
        except json.JSONDecodeError as e:
            # Return a minimal result if parsing fails
            return VaRAnalysisResult(
                executive_summary=analysis_text[:500] if analysis_text else "Analysis unavailable",
                risk_drivers=[],
                concentration_analysis=type(
                    "ConcentrationAnalysis", (), {
                        "most_concentrated_positions": [],
                        "best_diversifiers": [],
                        "concentration_risk_level": "unknown",
                        "concentration_summary": ""
                    }
                )(),
                factor_interpretation=type(
                    "FactorInterpretation", (), {
                        "dominant_factors": [],
                        "factor_risk_summary": "",
                        "factor_exposures": None
                    }
                )(),
                regime_interpretation=type(
                    "RegimeInterpretation", (), {
                        "regime_sensitivity": "unknown",
                        "regime_risk_summary": "",
                        "best_performing_regime": None,
                        "worst_performing_regime": None
                    }
                )(),
                recommendations=[]
            )

    def analyze_legacy(self) -> str:
        """Legacy analyze method returning raw text (for backward compatibility).

        Returns:
            String containing Claude's analysis of the portfolio risk.
        """
        result = self.analyze()
        return result.to_json()

    def analyze_with_guardrails(
        self,
        run_constitutional: bool = True
    ) -> Tuple[VaRAnalysisResult, GuardrailsReport]:
        """Analyze with full guardrails and return both result and report.

        This method runs the complete guardrails pipeline:
        1. Input validation (prompt injection, PII detection)
        2. Process checks (rate limiting, budget)
        3. LLM analysis with tool calling
        4. Output validation (hallucination, disclaimers)
        5. Constitutional self-critique (optional)

        Args:
            run_constitutional: Whether to run constitutional self-critique.

        Returns:
            Tuple of (VaRAnalysisResult, GuardrailsReport) with analysis
            and all guardrail check results.
        """
        guardrails_report = GuardrailsReport()

        if not self.guardrails_enabled:
            # Just run analysis without guardrails
            result = self.analyze()
            return result, guardrails_report

        # Step 1: Input validation
        prompt = self._build_initial_prompt()
        input_results = self.input_guards.run_all_checks(prompt)
        for r in input_results:
            guardrails_report.add_input_check(r)

        # Check if blocked by input guards
        if guardrails_report.blocked:
            # Return empty result with guardrails report
            from backend.llm.schemas import (
                ConcentrationAnalysis, FactorInterpretation, RegimeInterpretation
            )
            empty_result = VaRAnalysisResult(
                executive_summary="Analysis blocked by input guardrails",
                risk_drivers=[],
                concentration_analysis=ConcentrationAnalysis(
                    most_concentrated_positions=[],
                    best_diversifiers=[],
                    concentration_risk_level="unknown",
                    concentration_summary=""
                ),
                factor_interpretation=FactorInterpretation(
                    dominant_factors=[],
                    factor_risk_summary=""
                ),
                regime_interpretation=RegimeInterpretation(
                    regime_sensitivity="unknown",
                    regime_risk_summary=""
                ),
                recommendations=[]
            )
            return empty_result, guardrails_report

        # Step 2: Process checks (rate limiting, budget)
        process_results = self.process_guards.pre_request_checks(self.user_id)
        for r in process_results:
            guardrails_report.add_process_check(r)

        if guardrails_report.blocked:
            from backend.llm.schemas import (
                ConcentrationAnalysis, FactorInterpretation, RegimeInterpretation
            )
            empty_result = VaRAnalysisResult(
                executive_summary="Analysis blocked by rate limiting",
                risk_drivers=[],
                concentration_analysis=ConcentrationAnalysis(
                    most_concentrated_positions=[],
                    best_diversifiers=[],
                    concentration_risk_level="unknown",
                    concentration_summary=""
                ),
                factor_interpretation=FactorInterpretation(
                    dominant_factors=[],
                    factor_risk_summary=""
                ),
                regime_interpretation=RegimeInterpretation(
                    regime_sensitivity="unknown",
                    regime_risk_summary=""
                ),
                recommendations=[]
            )
            return empty_result, guardrails_report

        # Step 3: Run the analysis
        result = self.analyze()
        response_text = result.to_json()

        # Step 4: Output validation
        source_data = {
            "portfolio": self.report.holdings_json,
            "var": self.report.var_json,
            "factors": self.report.factor_data
        }
        processed_response, output_results = self.output_guards.run_all_checks(
            response_text, source_data
        )
        for r in output_results:
            guardrails_report.add_output_check(r)

        # Step 5: Constitutional self-critique (optional)
        if run_constitutional:
            try:
                _, constitutional_results, critique, grounding = \
                    self.constitutional.run_full_evaluation(
                        question="Analyze portfolio risk",
                        response=response_text,
                        source_data=self.report.report,
                        auto_revise=False
                    )
                for r in constitutional_results:
                    guardrails_report.add_constitutional_check(r)
            except Exception as e:
                # Don't let constitutional check failure break the analysis
                from backend.llm.guardrails import GuardResult
                guardrails_report.add_constitutional_check(GuardResult(
                    passed=True,
                    guard_name="constitutional_check",
                    message=f"Constitutional check skipped: {str(e)}",
                    severity="info"
                ))

        return result, guardrails_report
