"""Chat service for conversational portfolio Q&A with Claude AI.

This module provides functions for streaming chat conversations with Claude
about portfolio risk analysis, with the portfolio report as context.
Supports tool calling for real-time data lookup during conversations,
comprehensive guardrails for safety and reliability, and prompt caching
for cost optimization.
"""

import os
import json
from typing import List, Dict, Generator, Any, Optional, Tuple
from datetime import datetime
import anthropic

from backend.reporting.portfolio_report import PortfolioReport
from backend.llm.tools import TOOLS, execute_tool, set_portfolio_context
from backend.llm.guardrails import GuardrailsReport, GuardResult
from backend.llm.guardrails.input_guards import InputGuardrails
from backend.llm.guardrails.output_guards import OutputGuardrails
from backend.llm.guardrails.process_guards import ProcessGuardrails
from backend.llm.caching import (
    CachedClient,
    CacheConfig,
    CacheMetrics,
    CacheStats,
    get_session_tracker,
    record_cache_metrics,
    get_session_stats,
)


# Session-level cached client for reuse across calls
_cached_client: Optional[CachedClient] = None


def get_cached_client() -> CachedClient:
    """Get or create the session-level cached client.

    Returns:
        CachedClient instance for the session.
    """
    global _cached_client
    if _cached_client is None:
        _cached_client = CachedClient(
            config=CacheConfig(
                enabled=True,
                cache_system_prompt=True,
                cache_tools=True,
            )
        )
    return _cached_client


def get_system_prompt(report: PortfolioReport) -> str:
    """Build system prompt with portfolio context for Claude.

    Args:
        report: PortfolioReport containing all analysis data.

    Returns:
        Formatted system prompt string with portfolio data embedded.
    """
    return f"""You are a senior risk analyst helping a portfolio manager understand their portfolio risk.

Today is {datetime.today().strftime('%Y-%m-%d')}.

You have access to tools to look up additional information:
- lookup_stock_info: Get company sector, industry, and description for any ticker
- get_historical_prices: Get price data around a specific date
- get_market_news: Get recent news headlines for any ticker
- find_portfolio_diversifiers: Find stocks that would be good diversifiers for the portfolio based on lowest correlation and regime performance. Use this when asked about recommendations, stocks to add, or diversification opportunities.

You have access to the following portfolio data:

{report.report}

Answer questions about this portfolio accurately and concisely. Focus on:
- Risk metrics (VaR, Expected Shortfall)
- Position concentration and diversification
- Factor exposures
- Regime behavior

Use the available tools when you need additional information about stocks,
historical prices, or market news. When asked about stock recommendations or
diversification opportunities, use the find_portfolio_diversifiers tool to
analyze potential additions. After getting diversification candidates, provide
insightful commentary on WHY each stock is a good diversifier based on:
- Its low correlation to the portfolio
- How it performs in different market regimes (especially in bear markets or high volatility)
- What sector/industry diversification it provides

If you're unsure about something, say so. Do not fabricate data."""


def stream_chat_response(
    messages: List[Dict[str, str]],
    report: PortfolioReport
) -> Generator[str, None, None]:
    """Stream a chat response from Claude for the given conversation.

    Handles tool calls during the conversation, executing them and
    continuing until Claude provides a final text response.

    Args:
        messages: List of conversation messages with 'role' and 'content'.
        report: PortfolioReport providing context for the conversation.

    Yields:
        Text chunks as they stream from Claude's response.
    """
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        yield "Error: CLAUDE_API_KEY environment variable not set."
        return

    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = get_system_prompt(report)

    # Make a copy of messages to avoid modifying the original
    conversation = list(messages)

    max_tool_iterations = 5
    iteration = 0

    while iteration < max_tool_iterations:
        # Try streaming first for the final response
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system_prompt,
                tools=TOOLS,
                messages=conversation
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Execute tools
                tool_results = []
                tool_names_used = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_names_used.append(block.name)
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

                # Yield status message
                yield f"[Looking up: {', '.join(tool_names_used)}...]\n\n"

                # Add to conversation and continue
                conversation.append({
                    "role": "assistant",
                    "content": response.content
                })
                conversation.append({
                    "role": "user",
                    "content": tool_results
                })

                iteration += 1
            else:
                # No more tool calls, stream the final response
                # We need to make another call with streaming
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=conversation
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                break

        except Exception as e:
            yield f"Error: {str(e)}"
            break


def chat_response_with_tools(
    messages: List[Dict[str, str]],
    report: PortfolioReport
) -> str:
    """Get a complete chat response with tool support (non-streaming).

    Handles tool calls during the conversation, executing them and
    continuing until Claude provides a final text response.

    Args:
        messages: List of conversation messages with 'role' and 'content'.
        report: PortfolioReport providing context for the conversation.

    Returns:
        Complete response text from Claude.
    """
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        return "Error: CLAUDE_API_KEY environment variable not set."

    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = get_system_prompt(report)

    conversation = list(messages)
    max_tool_iterations = 5
    iteration = 0

    while iteration < max_tool_iterations:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system_prompt,
            tools=TOOLS,
            messages=conversation
        )

        if response.stop_reason == "tool_use":
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

            conversation.append({
                "role": "assistant",
                "content": response.content
            })
            conversation.append({
                "role": "user",
                "content": tool_results
            })
            iteration += 1
        else:
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            break

    return "Unable to generate response after maximum tool iterations."


def get_initial_analysis(report: PortfolioReport) -> str:
    """Generate initial AI summary analysis of the portfolio.

    Sends portfolio data to Claude for a concise executive summary
    covering risk profile, top contributors, and key exposures.
    Uses tool calling to enrich with stock information.

    Args:
        report: PortfolioReport containing all analysis data.

    Returns:
        String containing Claude's executive summary (under 300 words).
    """
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        return "Error: CLAUDE_API_KEY environment variable not set."

    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{
        "role": "user",
        "content": f"""Analyze this portfolio and provide a brief executive summary covering:
1. Overall risk profile (VaR interpretation)
2. Top 3 risk contributors
3. Key factor exposures
4. Regime sensitivity

You may use the available tools to look up additional stock information if helpful.
Keep your response concise (under 300 words).

Portfolio Data:
{report.report}"""
    }]

    max_iterations = 3
    iteration = 0

    while iteration < max_iterations:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "tool_use":
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
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text.strip()
            break

    return "Unable to generate analysis."


# Guardrails-enabled chat functions

def chat_with_guardrails(
    user_message: str,
    messages: List[Dict[str, str]],
    report: PortfolioReport,
    user_id: str = "default"
) -> Tuple[str, GuardrailsReport]:
    """Chat with full guardrails protection.

    Runs input validation, rate limiting, and output checks around
    the chat response.

    Args:
        user_message: The new user message to respond to.
        messages: Previous conversation messages.
        report: PortfolioReport providing context.
        user_id: User identifier for rate limiting.

    Returns:
        Tuple of (response_text, GuardrailsReport).
    """
    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    guardrails_report = GuardrailsReport()

    # Initialize guardrails
    input_guards = InputGuardrails()
    output_guards = OutputGuardrails()
    process_guards = ProcessGuardrails()

    # Step 1: Input validation
    input_results = input_guards.run_all_checks(user_message)
    for r in input_results:
        guardrails_report.add_input_check(r)

    if guardrails_report.blocked:
        blocked_msg = "I cannot process this request due to safety guardrails."
        return blocked_msg, guardrails_report

    # Step 2: Process checks
    process_results = process_guards.pre_request_checks(user_id)
    for r in process_results:
        guardrails_report.add_process_check(r)

    if guardrails_report.blocked:
        rate_msg = "Rate limit exceeded. Please try again later."
        return rate_msg, guardrails_report

    # Step 3: Get response
    full_messages = messages + [{"role": "user", "content": user_message}]
    response = chat_response_with_tools(full_messages, report)

    # Step 4: Output validation
    source_data = {
        "portfolio": report.holdings_json,
        "var": report.var_json,
        "factors": report.factor_data
    }
    processed_response, output_results = output_guards.run_all_checks(
        response, source_data
    )
    for r in output_results:
        guardrails_report.add_output_check(r)

    return processed_response, guardrails_report


def stream_chat_with_guardrails(
    user_message: str,
    messages: List[Dict[str, str]],
    report: PortfolioReport,
    user_id: str = "default"
) -> Generator[Tuple[str, Optional[GuardrailsReport]], None, None]:
    """Stream chat response with guardrails.

    Yields response chunks during streaming, with a final guardrails
    report at the end.

    Args:
        user_message: The new user message to respond to.
        messages: Previous conversation messages.
        report: PortfolioReport providing context.
        user_id: User identifier for rate limiting.

    Yields:
        Tuples of (text_chunk, None) during streaming,
        then ("", GuardrailsReport) at the end.
    """
    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    guardrails_report = GuardrailsReport()

    # Initialize guardrails
    input_guards = InputGuardrails()
    process_guards = ProcessGuardrails()
    output_guards = OutputGuardrails()

    # Step 1: Input validation
    input_results = input_guards.run_all_checks(user_message)
    for r in input_results:
        guardrails_report.add_input_check(r)

    if guardrails_report.blocked:
        yield "I cannot process this request due to safety guardrails.", None
        yield "", guardrails_report
        return

    # Step 2: Process checks
    process_results = process_guards.pre_request_checks(user_id)
    for r in process_results:
        guardrails_report.add_process_check(r)

    if guardrails_report.blocked:
        yield "Rate limit exceeded. Please try again later.", None
        yield "", guardrails_report
        return

    # Step 3: Stream response and collect full text
    full_messages = messages + [{"role": "user", "content": user_message}]
    full_response = ""

    for chunk in stream_chat_response(full_messages, report):
        full_response += chunk
        yield chunk, None

    # Step 4: Output validation (on complete response)
    source_data = {
        "portfolio": report.holdings_json,
        "var": report.var_json,
        "factors": report.factor_data
    }
    _, output_results = output_guards.run_all_checks(full_response, source_data)
    for r in output_results:
        guardrails_report.add_output_check(r)

    # Yield final guardrails report
    yield "", guardrails_report


def get_guardrails_summary(report: GuardrailsReport) -> Dict[str, Any]:
    """Get a summary of guardrails results for display.

    Args:
        report: GuardrailsReport from a guarded operation.

    Returns:
        Dictionary with summary suitable for UI display.
    """
    return {
        "status": "blocked" if report.blocked else ("passed" if report.overall_passed else "warnings"),
        "total_checks": (
            len(report.input_checks) +
            len(report.output_checks) +
            len(report.constitutional_checks) +
            len(report.process_checks)
        ),
        "warnings": [
            {"guard": r.guard_name, "message": r.message}
            for r in report.get_warnings()
        ],
        "errors": [
            {"guard": r.guard_name, "message": r.message}
            for r in report.get_errors()
        ],
        "checks": {
            "input": [{"name": r.guard_name, "passed": r.passed, "severity": r.severity}
                     for r in report.input_checks],
            "output": [{"name": r.guard_name, "passed": r.passed, "severity": r.severity}
                      for r in report.output_checks],
            "process": [{"name": r.guard_name, "passed": r.passed, "severity": r.severity}
                       for r in report.process_checks],
            "constitutional": [{"name": r.guard_name, "passed": r.passed, "severity": r.severity}
                              for r in report.constitutional_checks],
        }
    }


# ============================================================================
# Prompt Caching Functions
# ============================================================================

def chat_response_with_caching(
    messages: List[Dict[str, str]],
    report: PortfolioReport
) -> Tuple[str, CacheMetrics]:
    """Get a chat response with prompt caching enabled.

    Uses Anthropic's prompt caching to cache the large portfolio context,
    reducing costs by up to 90% for subsequent calls.

    Args:
        messages: List of conversation messages with 'role' and 'content'.
        report: PortfolioReport providing context for the conversation.

    Returns:
        Tuple of (response_text, CacheMetrics).
    """
    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    client = get_cached_client()
    system_prompt = get_system_prompt(report)

    conversation = list(messages)
    max_tool_iterations = 5
    iteration = 0
    last_metrics: Optional[CacheMetrics] = None

    while iteration < max_tool_iterations:
        response, metrics = client.create_message_with_cache(
            system_prompt=system_prompt,
            messages=conversation,
            tools=TOOLS,
            max_tokens=2000,
        )
        last_metrics = metrics

        if response.stop_reason == "tool_use":
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

            conversation.append({
                "role": "assistant",
                "content": response.content
            })
            conversation.append({
                "role": "user",
                "content": tool_results
            })
            iteration += 1
        else:
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text, last_metrics
            break

    return "Unable to generate response after maximum tool iterations.", last_metrics


def stream_chat_with_caching(
    messages: List[Dict[str, str]],
    report: PortfolioReport
) -> Generator[Tuple[str, Optional[CacheMetrics]], None, None]:
    """Stream a chat response with prompt caching enabled.

    Yields text chunks during streaming, with cache metrics at the end.

    Args:
        messages: List of conversation messages with 'role' and 'content'.
        report: PortfolioReport providing context for the conversation.

    Yields:
        Tuples of (text_chunk, None) during streaming,
        then ("", CacheMetrics) at the end.
    """
    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    client = get_cached_client()
    system_prompt = get_system_prompt(report)

    conversation = list(messages)
    max_tool_iterations = 5
    iteration = 0

    while iteration < max_tool_iterations:
        # First check if tools are needed (non-streaming)
        response, metrics = client.create_message_with_cache(
            system_prompt=system_prompt,
            messages=conversation,
            tools=TOOLS,
            max_tokens=2000,
        )

        if response.stop_reason == "tool_use":
            # Execute tools
            tool_results = []
            tool_names_used = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_names_used.append(block.name)
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

            yield f"[Looking up: {', '.join(tool_names_used)}...]\n\n", None

            conversation.append({
                "role": "assistant",
                "content": response.content
            })
            conversation.append({
                "role": "user",
                "content": tool_results
            })
            iteration += 1
        else:
            # Stream the final response
            for chunk, chunk_metrics in client.stream_message_with_cache(
                system_prompt=system_prompt,
                messages=conversation,
                tools=TOOLS,
                max_tokens=2000,
            ):
                if chunk_metrics:
                    yield "", chunk_metrics
                else:
                    yield chunk, None
            break


def chat_with_caching_and_guardrails(
    user_message: str,
    messages: List[Dict[str, str]],
    report: PortfolioReport,
    user_id: str = "default"
) -> Tuple[str, GuardrailsReport, CacheMetrics]:
    """Chat with both guardrails protection and prompt caching.

    Combines guardrails for safety with prompt caching for cost efficiency.

    Args:
        user_message: The new user message to respond to.
        messages: Previous conversation messages.
        report: PortfolioReport providing context.
        user_id: User identifier for rate limiting.

    Returns:
        Tuple of (response_text, GuardrailsReport, CacheMetrics).
    """
    # Set portfolio context for tools that need it
    set_portfolio_context(report.portfolio)

    guardrails_report = GuardrailsReport()

    # Initialize guardrails
    input_guards = InputGuardrails()
    output_guards = OutputGuardrails()
    process_guards = ProcessGuardrails()

    # Step 1: Input validation
    input_results = input_guards.run_all_checks(user_message)
    for r in input_results:
        guardrails_report.add_input_check(r)

    if guardrails_report.blocked:
        blocked_msg = "I cannot process this request due to safety guardrails."
        empty_metrics = CacheMetrics()
        return blocked_msg, guardrails_report, empty_metrics

    # Step 2: Process checks
    process_results = process_guards.pre_request_checks(user_id)
    for r in process_results:
        guardrails_report.add_process_check(r)

    if guardrails_report.blocked:
        rate_msg = "Rate limit exceeded. Please try again later."
        empty_metrics = CacheMetrics()
        return rate_msg, guardrails_report, empty_metrics

    # Step 3: Get response with caching
    full_messages = messages + [{"role": "user", "content": user_message}]
    response, cache_metrics = chat_response_with_caching(full_messages, report)

    # Step 4: Output validation
    source_data = {
        "portfolio": report.holdings_json,
        "var": report.var_json,
        "factors": report.factor_data
    }
    processed_response, output_results = output_guards.run_all_checks(
        response, source_data
    )
    for r in output_results:
        guardrails_report.add_output_check(r)

    return processed_response, guardrails_report, cache_metrics


def get_cache_stats() -> Dict[str, Any]:
    """Get session-wide cache statistics.

    Returns:
        Dictionary with cache performance metrics.
    """
    return get_session_stats()


def get_cache_status_display() -> Dict[str, Any]:
    """Get cache status formatted for UI display.

    Returns:
        Dictionary with user-friendly cache status.
    """
    stats = get_session_stats()
    client = get_cached_client()
    last = client.get_last_metrics()

    return {
        "enabled": True,
        "session": {
            "total_calls": stats.get("total_calls", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", "0%"),
            "estimated_savings": stats.get("estimated_savings", "0%"),
            "total_cached_tokens": stats.get("total_cached_tokens", 0),
        },
        "last_call": {
            "cache_hit": last.cache_hit if last else False,
            "cached_tokens": last.cache_read_tokens if last else 0,
            "savings_percent": f"{last.estimated_savings_percent:.1f}%" if last else "0%",
        } if last else None
    }
