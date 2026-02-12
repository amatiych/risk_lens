"""13F Filing Agent for automated portfolio import from SEC filings.

This agent orchestrates the workflow of:
1. Searching for institutional investors
2. Downloading their latest 13F filings
3. Converting to portfolio format
4. Running risk analysis

Uses the existing LLM provider abstraction for multi-provider support.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from backend.llm.providers import LLMConfig, get_provider, LLMProvider
from backend.llm.tools_13f import TOOLS_13F, execute_13f_tool


@dataclass
class Agent13FResult:
    """Result from 13F agent workflow."""

    success: bool
    portfolio_name: Optional[str] = None
    portfolio_csv: Optional[str] = None
    analysis_results: Optional[Dict[str, Any]] = None
    filer_info: Optional[Dict[str, Any]] = None
    filing_info: Optional[Dict[str, Any]] = None
    holdings_count: int = 0
    mapped_tickers: int = 0
    unmapped_cusips: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    status_messages: List[str] = field(default_factory=list)


AGENT_SYSTEM_PROMPT = """You are an SEC 13F filing research agent. Your task is to help users analyze institutional investor portfolios from SEC 13F filings.

You have access to these tools:
- search_sec_filer: Find an institutional investor by name. Returns their CIK (Central Index Key).
- get_latest_13f_filing: Get the most recent 13F-HR filing for a CIK. Returns the accession number and filing date.
- download_13f_holdings: Download the holdings from a 13F filing. Returns positions with CUSIPs and share counts.
- convert_13f_to_portfolio: Convert 13F holdings (CUSIPs) to portfolio format (tickers). Handles CUSIP-to-ticker mapping.
- run_13f_analysis: Run the converted portfolio through the Risk Lens analysis pipeline.

WORKFLOW - Follow these steps in order:
1. Use search_sec_filer to find the institutional investor the user mentioned
2. If multiple results, pick the most relevant one (usually the first match)
3. Use get_latest_13f_filing with the CIK to get the latest 13F filing
4. Use download_13f_holdings with the CIK and accession_number to get holdings
5. Use convert_13f_to_portfolio to convert holdings to ticker format (use top_n if user specified a limit)
6. Use run_13f_analysis to run the risk analysis on the portfolio

IMPORTANT:
- Always complete all 6 steps to give the user a full analysis
- If a step fails, explain the issue clearly and try to recover if possible
- After download_13f_holdings, report how many positions were found
- After convert_13f_to_portfolio, report how many were successfully mapped
- At the end, summarize: filer name, filing date, positions analyzed, and key risk metrics

Do NOT skip steps. Execute each tool in sequence."""


class Agent13F:
    """Agent for 13F filing workflow."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        max_iterations: int = 15,
    ):
        """Initialize the 13F agent.

        Args:
            config: LLM configuration. Defaults to using session provider.
            max_iterations: Maximum tool execution iterations.
        """
        self.config = config or LLMConfig()
        self.provider: LLMProvider = get_provider(self.config)
        self.max_iterations = max_iterations

    def run(
        self, user_query: str, top_n: Optional[int] = None
    ) -> Generator[str, None, Agent13FResult]:
        """Run the 13F agent workflow.

        Yields status updates during execution.
        Returns final Agent13FResult when complete.

        Args:
            user_query: User's request (e.g., "Find Berkshire Hathaway's latest 13F")
            top_n: Optional limit on number of holdings to analyze.

        Yields:
            Status message strings during execution.

        Returns:
            Agent13FResult with complete workflow results.
        """
        result = Agent13FResult(success=False)
        result.status_messages = []

        # Build the initial message
        if top_n:
            user_message = f"{user_query}\n\nLimit the analysis to the top {top_n} holdings by value."
        else:
            user_message = user_query

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        yield f"Starting 13F workflow: {user_query}"
        result.status_messages.append(f"Query: {user_query}")

        iteration = 0

        while iteration < self.max_iterations:
            try:
                response = self.provider.create_message(
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOLS_13F,
                    max_tokens=4000,
                )

                # Check if LLM wants to use tools
                if response.stop_reason == "tool_use" and response.tool_calls:
                    tool_results = []

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.name
                        tool_args = tool_call.arguments

                        status = f"Executing: {tool_name}"
                        yield status
                        result.status_messages.append(status)

                        try:
                            tool_result = execute_13f_tool(tool_name, tool_args)

                            # Extract useful info from tool results
                            self._update_result_from_tool(
                                result, tool_name, tool_result
                            )

                            tool_results.append(
                                self.provider.format_tool_result(
                                    tool_call.id, json.dumps(tool_result)
                                )
                            )

                            # Yield progress based on tool results
                            if tool_result.get("success"):
                                if tool_name == "search_sec_filer":
                                    count = tool_result.get("results_count", 0)
                                    yield f"Found {count} matching filer(s)"
                                elif tool_name == "get_latest_13f_filing":
                                    filing = tool_result.get("latest_filing", {})
                                    yield f"Found filing from {filing.get('filing_date', 'unknown date')}"
                                elif tool_name == "download_13f_holdings":
                                    count = tool_result.get("total_positions", 0)
                                    yield f"Downloaded {count} holdings"
                                elif tool_name == "convert_13f_to_portfolio":
                                    stats = tool_result.get("statistics", {})
                                    yield f"Mapped {stats.get('successful_mappings', 0)} of {stats.get('total_holdings_input', 0)} holdings to tickers"
                                elif tool_name == "run_13f_analysis":
                                    yield "Analysis complete"
                            else:
                                error = tool_result.get("error", "Unknown error")
                                yield f"Warning: {tool_name} - {error}"

                        except Exception as e:
                            error_result = {"error": str(e)}
                            tool_results.append(
                                self.provider.format_tool_result(
                                    tool_call.id,
                                    json.dumps(error_result),
                                    is_error=True,
                                )
                            )
                            yield f"Error in {tool_name}: {str(e)}"

                    # Add assistant message with tool calls to conversation
                    messages.append(
                        self.provider.format_assistant_message(
                            response.content, response.tool_calls
                        )
                    )
                    # Add tool results
                    messages.append({"role": "user", "content": tool_results})

                    iteration += 1
                else:
                    # No more tool calls, workflow complete
                    if response.content:
                        result.conversation_history = messages.copy()
                        result.success = (
                            result.portfolio_csv is not None
                            and result.analysis_results is not None
                        )

                        yield "Workflow complete"
                        return result

                    break

            except Exception as e:
                result.error_message = str(e)
                yield f"Error: {str(e)}"
                return result

        # Max iterations reached
        result.error_message = "Maximum iterations reached without completing workflow"
        yield "Warning: Maximum iterations reached"
        return result

    def run_sync(
        self, user_query: str, top_n: Optional[int] = None
    ) -> Agent13FResult:
        """Synchronous version of run().

        Args:
            user_query: User's request.
            top_n: Optional limit on holdings.

        Returns:
            Agent13FResult with workflow results.
        """
        result = None
        for status in self.run(user_query, top_n):
            pass  # Consume generator
        # The generator returns the result when it completes
        # We need to get it by running through the generator
        gen = self.run(user_query, top_n)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        return result if result else Agent13FResult(
            success=False, error_message="No result returned"
        )

    def _update_result_from_tool(
        self,
        result: Agent13FResult,
        tool_name: str,
        tool_result: Dict[str, Any],
    ) -> None:
        """Update the Agent13FResult with data from tool execution.

        Args:
            result: The result object to update.
            tool_name: Name of the tool that was executed.
            tool_result: Result from the tool execution.
        """
        if not tool_result.get("success"):
            return

        if tool_name == "search_sec_filer":
            filers = tool_result.get("filers", [])
            if filers:
                result.filer_info = filers[0]

        elif tool_name == "get_latest_13f_filing":
            result.filing_info = tool_result.get("latest_filing")

        elif tool_name == "download_13f_holdings":
            result.holdings_count = tool_result.get("total_positions", 0)

        elif tool_name == "convert_13f_to_portfolio":
            result.portfolio_csv = tool_result.get("portfolio_csv")
            stats = tool_result.get("statistics", {})
            result.mapped_tickers = stats.get("successful_mappings", 0)

            unmapped = tool_result.get("unmapped_holdings", [])
            result.unmapped_cusips = [h.get("cusip", "") for h in unmapped]

            # Generate portfolio name
            if result.filer_info and result.filing_info:
                filer_name = result.filer_info.get("name", "Unknown")
                filing_date = result.filing_info.get("report_date", "")
                result.portfolio_name = f"{filer_name} 13F ({filing_date})"

        elif tool_name == "run_13f_analysis":
            result.analysis_results = tool_result
