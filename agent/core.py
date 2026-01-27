"""Core Portfolio Analysis Agent.

This module provides the main PortfolioAgent class that orchestrates
AI-powered portfolio analysis using tools and generates comprehensive reports.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tool_registry import ToolRegistry
from backend.llm.providers.base import LLMProvider, ProviderResponse
from backend.llm.providers.claude_provider import ClaudeProvider
from backend.llm.providers.openai_provider import OpenAIProvider


@dataclass
class AgentConfig:
    """Configuration for the Portfolio Agent.

    Attributes:
        provider: LLM provider to use ('claude' or 'openai').
        max_iterations: Maximum tool call iterations per request.
        verbose: Whether to print debug information.
        report_output_dir: Directory for saving reports.
    """
    provider: str = "claude"
    max_iterations: int = 20
    verbose: bool = False
    report_output_dir: str = "./reports"


@dataclass
class ToolExecution:
    """Record of a tool execution.

    Attributes:
        tool_name: Name of the executed tool.
        arguments: Arguments passed to the tool.
        result: Result from the tool execution.
        timestamp: When the tool was executed.
    """
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResponse:
    """Response from the agent.

    Attributes:
        content: Final text response from the agent.
        tool_executions: List of tools that were executed.
        charts_generated: List of chart file paths generated.
        report_path: Path to generated report if any.
    """
    content: str
    tool_executions: List[ToolExecution] = field(default_factory=list)
    charts_generated: List[str] = field(default_factory=list)
    report_path: Optional[str] = None


SYSTEM_PROMPT = """You are a Portfolio Risk Analysis Agent. Your role is to analyze investment portfolios
and provide comprehensive risk assessments using the available tools.

## Your Capabilities
You have access to tools for:
1. **Portfolio Management**: Load portfolios, view holdings, execute trades
2. **Risk Analytics**: Calculate VaR, factor exposures, regime performance, PCA analysis
3. **Market Data**: Look up stock info, historical prices, news, find diversifiers
4. **Chart Generation**: Create visualizations for reports

## How to Work
1. When given a portfolio ID, first load the portfolio using the `load_portfolio` tool
2. Run appropriate analyses based on the user's request
3. Generate charts to visualize findings
4. Provide clear, actionable insights

## Response Guidelines
- Always explain your findings in plain language
- Highlight key risks and opportunities
- Support conclusions with data from the tools
- When generating reports, include relevant charts
- Be specific about numbers and percentages

## Important Notes
- You must load a portfolio before running any analysis
- Factor and regime analyses require market data which may take time to fetch
- Charts are saved to files or returned as base64 for embedding in reports
"""


class PortfolioAgent:
    """AI-powered portfolio analysis agent.

    Uses an LLM with tools to analyze portfolios and generate reports.

    Attributes:
        config: Agent configuration.
        provider: LLM provider instance.
        registry: Tool registry.
        messages: Conversation history.
        tool_executions: History of tool executions.

    Example:
        agent = PortfolioAgent()
        response = agent.analyze(102, "Give me a full risk analysis")
        print(response.content)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the portfolio agent.

        Args:
            config: Optional agent configuration.
        """
        self.config = config or AgentConfig()
        self.provider = self._create_provider()
        self.registry = ToolRegistry()
        self.messages: List[Dict[str, Any]] = []
        self.tool_executions: List[ToolExecution] = []

        # Import tools to trigger registration
        import agent.tools  # noqa: F401

    def _create_provider(self) -> LLMProvider:
        """Create the LLM provider based on config."""
        if self.config.provider == "openai":
            return OpenAIProvider()
        return ClaudeProvider()

    def analyze(
        self,
        portfolio_id: int,
        request: str,
        generate_report: bool = False
    ) -> AgentResponse:
        """Analyze a portfolio based on the request.

        Args:
            portfolio_id: ID of the portfolio to analyze.
            request: Natural language request describing desired analysis.
            generate_report: Whether to generate a markdown report.

        Returns:
            AgentResponse with analysis results.
        """
        # Clear previous state
        self.messages = []
        self.tool_executions = []
        self.registry.clear_context()

        # Build the user message with portfolio context
        user_message = f"""Please analyze portfolio ID {portfolio_id}.

Request: {request}

Start by loading the portfolio, then perform the requested analysis."""

        self.messages.append({"role": "user", "content": user_message})

        # Run the agent loop
        final_content = self._run_agent_loop()

        # Generate report if requested
        report_path = None
        charts = []
        if generate_report:
            report_path, charts = self._generate_report(portfolio_id, final_content)

        return AgentResponse(
            content=final_content,
            tool_executions=self.tool_executions,
            charts_generated=charts,
            report_path=report_path
        )

    def chat(self, message: str) -> AgentResponse:
        """Continue the conversation with a follow-up message.

        Args:
            message: User's follow-up message.

        Returns:
            AgentResponse with the response.
        """
        self.messages.append({"role": "user", "content": message})
        final_content = self._run_agent_loop()

        return AgentResponse(
            content=final_content,
            tool_executions=self.tool_executions[-10:]  # Last 10 executions
        )

    def _run_agent_loop(self) -> str:
        """Run the agent loop until completion or max iterations.

        Returns:
            Final text content from the agent.
        """
        tools = self.registry.get_tool_definitions()

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Call the LLM
            response = self.provider.create_message(
                system_prompt=SYSTEM_PROMPT,
                messages=self.messages,
                tools=tools,
                max_tokens=4096
            )

            # Handle the response
            if response.stop_reason == "end_turn":
                return response.content or ""

            if response.stop_reason == "tool_use" and response.tool_calls:
                # Execute tools and continue
                self._handle_tool_calls(response)
            else:
                # Unexpected stop reason
                return response.content or ""

        # Max iterations reached
        return "Analysis incomplete - maximum iterations reached."

    def _handle_tool_calls(self, response: ProviderResponse) -> None:
        """Handle tool calls from the LLM response.

        Args:
            response: LLM response containing tool calls.
        """
        # Add assistant message to history
        assistant_msg = self.provider.format_assistant_message(
            response.content, response.tool_calls
        )
        self.messages.append(assistant_msg)

        # Execute each tool and collect results
        tool_results = []
        for tool_call in response.tool_calls:
            if self.config.verbose:
                print(f"  Executing: {tool_call.name}({tool_call.arguments})")

            try:
                result = self.registry.execute(tool_call.name, tool_call.arguments)
            except Exception as e:
                result = {"error": str(e)}

            # Record execution
            self.tool_executions.append(ToolExecution(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=result
            ))

            if self.config.verbose:
                print(f"  Result: {json.dumps(result, indent=2)[:200]}...")

            # Format tool result
            tool_result = self.provider.format_tool_result(
                tool_call.id,
                json.dumps(result),
                is_error="error" in result
            )
            tool_results.append(tool_result)

        # Add tool results to messages
        self.messages.append({"role": "user", "content": tool_results})

    def _generate_report(
        self,
        portfolio_id: int,
        analysis_content: str
    ) -> tuple[Optional[str], List[str]]:
        """Generate a markdown report with charts.

        Args:
            portfolio_id: Portfolio ID for naming.
            analysis_content: Main analysis content.

        Returns:
            Tuple of (report_path, list_of_chart_paths).
        """
        output_dir = Path(self.config.report_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"portfolio_{portfolio_id}_report_{timestamp}"
        report_path = output_dir / f"{report_name}.md"
        charts_dir = output_dir / f"{report_name}_charts"
        charts_dir.mkdir(exist_ok=True)

        charts = []
        portfolio = self.registry.get_context("portfolio")
        portfolio_name = portfolio.name if portfolio else f"Portfolio {portfolio_id}"

        # Generate charts
        chart_configs = [
            ("holdings", {"chart_type": "bar", "top_n": 10}),
            ("correlation", {"top_n": 10}),
            ("factor", {}),
            ("regime", {}),
            ("var", {"confidence_level": 0.95}),
            ("pca", {"num_components": 10}),
        ]

        for chart_name, params in chart_configs:
            try:
                chart_path = str(charts_dir / f"{chart_name}.png")
                params["output_path"] = chart_path

                if chart_name == "holdings":
                    from agent.tools.chart_tools import generate_holdings_chart
                    result = generate_holdings_chart(_context={"portfolio": portfolio}, **params)
                elif chart_name == "correlation":
                    from agent.tools.chart_tools import generate_correlation_heatmap
                    result = generate_correlation_heatmap(_context={"portfolio": portfolio}, **params)
                elif chart_name == "factor":
                    from agent.tools.chart_tools import generate_factor_chart
                    result = generate_factor_chart(_context={"portfolio": portfolio}, **params)
                elif chart_name == "regime":
                    from agent.tools.chart_tools import generate_regime_chart
                    result = generate_regime_chart(_context={"portfolio": portfolio}, **params)
                elif chart_name == "var":
                    from agent.tools.chart_tools import generate_var_chart
                    result = generate_var_chart(_context={"portfolio": portfolio}, **params)
                elif chart_name == "pca":
                    from agent.tools.chart_tools import generate_pca_chart
                    result = generate_pca_chart(_context={"portfolio": portfolio}, **params)

                if "error" not in result:
                    charts.append(chart_path)
            except Exception as e:
                if self.config.verbose:
                    print(f"Failed to generate {chart_name} chart: {e}")

        # Build markdown report
        report_content = f"""# Portfolio Analysis Report: {portfolio_name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Portfolio ID:** {portfolio_id}

---

## Executive Summary

{analysis_content}

---

## Charts

"""
        for chart_path in charts:
            chart_name = Path(chart_path).stem.replace("_", " ").title()
            rel_path = Path(chart_path).relative_to(output_dir)
            report_content += f"### {chart_name}\n\n![{chart_name}]({rel_path})\n\n"

        report_content += """---

## Tool Execution Log

| Tool | Timestamp |
|------|-----------|
"""
        for exec in self.tool_executions:
            report_content += f"| {exec.tool_name} | {exec.timestamp.strftime('%H:%M:%S')} |\n"

        report_content += "\n---\n\n*Report generated by Portfolio Analysis Agent*\n"

        # Save report
        report_path.write_text(report_content)

        return str(report_path), charts

    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get list of available tools grouped by category.

        Returns:
            Dictionary mapping categories to tool names.
        """
        return self.registry.list_by_category()


def run_analysis(
    portfolio_id: int,
    request: str = "Provide a comprehensive risk analysis",
    generate_report: bool = True,
    provider: str = "claude",
    verbose: bool = False
) -> AgentResponse:
    """Convenience function to run a portfolio analysis.

    Args:
        portfolio_id: ID of the portfolio to analyze.
        request: Analysis request in natural language.
        generate_report: Whether to generate a markdown report.
        provider: LLM provider ('claude' or 'openai').
        verbose: Whether to print debug output.

    Returns:
        AgentResponse with analysis results.

    Example:
        response = run_analysis(102, "What are the main risks in this portfolio?")
        print(response.content)
    """
    config = AgentConfig(
        provider=provider,
        verbose=verbose
    )
    agent = PortfolioAgent(config)
    return agent.analyze(portfolio_id, request, generate_report)
