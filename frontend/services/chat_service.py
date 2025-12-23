"""Chat service for conversational portfolio Q&A with Claude AI.

This module provides functions for streaming chat conversations with Claude
about portfolio risk analysis, with the portfolio report as context.
Supports tool calling for real-time data lookup during conversations.
"""

import os
import json
from typing import List, Dict, Generator, Any, Optional
from datetime import datetime
import anthropic

from backend.reporting.portfolio_report import PortfolioReport
from backend.llm.tools import TOOLS, execute_tool


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

You have access to the following portfolio data:

{report.report}

Answer questions about this portfolio accurately and concisely. Focus on:
- Risk metrics (VaR, Expected Shortfall)
- Position concentration and diversification
- Factor exposures
- Regime behavior

Use the available tools when you need additional information about stocks,
historical prices, or market news. If you're unsure about something, say so.
Do not fabricate data."""


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
