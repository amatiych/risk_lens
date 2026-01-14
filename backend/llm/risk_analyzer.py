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


class RiskAnalyzer:
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

    def _build_system_prompt(self) -> str:
        """Build concise system prompt.

        Returns:
            System prompt string (cached separately from report data).
        """
        return f"""Senior hedge fund risk manager. Date: {datetime.today().strftime('%Y-%m-%d')}.
Analyze portfolio risk: drivers (marginal/incremental VaR), concentration, factors, regimes
If availbale provide factor diversification analysis based on PCA data. 
Use tools for stock info, historical prices, and news context. Use only factual data."""

    def _build_user_prompt(self) -> str:
        """Build user prompt with report data.

        Returns:
            User prompt with embedded report.
        """
        return f"""Analyze this portfolio risk report:

{self.report.report}

Identify top risk drivers, concentration risks, factor exposures, regime sensitivity, and recommendations."""

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

        Uses single-pass structured output with tool_choice for efficiency.
        Handles tool calls in a loop until Claude provides the final
        structured response via the var_analysis_result tool.

        Returns:
            VaRAnalysisResult containing structured analysis with risk drivers,
            concentration analysis, factor interpretation, regime insights,
            and recommendations.

        Raises:
            anthropic.APIError: If the API call fails.
            ValueError: If unable to parse structured response.
        """
        client = anthropic.Anthropic(api_key=self.api_key)

        # Build system prompt with cache control for efficiency
        system_prompt = [
            {
                "type": "text",
                "text": self._build_system_prompt(),
                "cache_control": {"type": "ephemeral"}
            }
        ]

        messages = [
            {"role": "user", "content": self._build_user_prompt()}
        ]

        # Combined tools: data lookup + structured output
        all_tools = TOOLS + [
            {
                "name": "var_analysis_result",
                "description": "Submit the final structured VaR analysis. Call this after gathering all necessary information.",
                "input_schema": VAR_ANALYSIS_SCHEMA["schema"]
            }
        ]

        iteration = 0

        while iteration < self.max_tool_iterations:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                system=system_prompt,
                tools=all_tools,
                messages=messages
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                tool_results = []
                final_result = None

                for block in response.content:
                    if block.type == "tool_use":
                        # Check if this is the final structured output
                        if block.name == "var_analysis_result":
                            final_result = VaRAnalysisResult.from_dict(block.input)
                        else:
                            # Execute data lookup tools
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

                # If we got the final result, return it
                if final_result:
                    return final_result

                # Otherwise continue the conversation with tool results
                if tool_results:
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
                # No tool use - force structured output
                break

        # Force final structured output if we haven't got it yet
        messages.append({
            "role": "user",
            "content": "Provide your analysis now using the var_analysis_result tool."
        })

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=system_prompt,
            tools=[{
                "name": "var_analysis_result",
                "description": "Submit the final structured VaR analysis",
                "input_schema": VAR_ANALYSIS_SCHEMA["schema"]
            }],
            tool_choice={"type": "tool", "name": "var_analysis_result"},
            messages=messages
        )

        # Extract from forced tool call
        for block in response.content:
            if block.type == "tool_use" and block.name == "var_analysis_result":
                return VaRAnalysisResult.from_dict(block.input)

        # Fallback if parsing fails
        return self._empty_result("Unable to generate structured analysis")

    def _empty_result(self, message: str) -> VaRAnalysisResult:
        """Create an empty result with error message.

        Args:
            message: Error message for executive summary.

        Returns:
            VaRAnalysisResult with empty fields.
        """
        from backend.llm.schemas import (
            ConcentrationAnalysis, FactorInterpretation, RegimeInterpretation
        )
        return VaRAnalysisResult(
            executive_summary=message,
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
            result = self.analyze()
            return result, guardrails_report

        # Step 1: Input validation
        prompt = self._build_user_prompt()
        input_results = self.input_guards.run_all_checks(prompt)
        for r in input_results:
            guardrails_report.add_input_check(r)

        if guardrails_report.blocked:
            return self._empty_result("Analysis blocked by input guardrails"), guardrails_report

        # Step 2: Process checks (rate limiting, budget)
        process_results = self.process_guards.pre_request_checks(self.user_id)
        for r in process_results:
            guardrails_report.add_process_check(r)

        if guardrails_report.blocked:
            return self._empty_result("Analysis blocked by rate limiting"), guardrails_report

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
                from backend.llm.guardrails import GuardResult
                guardrails_report.add_constitutional_check(GuardResult(
                    passed=True,
                    guard_name="constitutional_check",
                    message=f"Constitutional check skipped: {str(e)}",
                    severity="info"
                ))

        return result, guardrails_report
