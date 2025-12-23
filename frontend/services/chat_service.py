"""Chat service for conversational portfolio Q&A with Claude AI.

This module provides functions for streaming chat conversations with Claude
about portfolio risk analysis, with the portfolio report as context.
"""

import os
from typing import List, Dict, Generator
from datetime import datetime
import anthropic

from backend.reporting.portfolio_report import PortfolioReport


def get_system_prompt(report: PortfolioReport) -> str:
    """Build system prompt with portfolio context for Claude.

    Args:
        report: PortfolioReport containing all analysis data.

    Returns:
        Formatted system prompt string with portfolio data embedded.
    """
    return f"""You are a senior risk analyst helping a portfolio manager understand their portfolio risk.

Today is {datetime.today().strftime('%Y-%m-%d')}.

You have access to the following portfolio data:

{report.report}

Answer questions about this portfolio accurately and concisely. Focus on:
- Risk metrics (VaR, Expected Shortfall)
- Position concentration and diversification
- Factor exposures
- Regime behavior

If you're unsure about something, say so. Do not fabricate data."""


def stream_chat_response(
    messages: List[Dict[str, str]],
    report: PortfolioReport
) -> Generator[str, None, None]:
    """Stream a chat response from Claude for the given conversation.

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

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            yield text


def get_initial_analysis(report: PortfolioReport) -> str:
    """Generate initial AI summary analysis of the portfolio.

    Sends portfolio data to Claude for a concise executive summary
    covering risk profile, top contributors, and key exposures.

    Args:
        report: PortfolioReport containing all analysis data.

    Returns:
        String containing Claude's executive summary (under 300 words).
    """
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        return "Error: CLAUDE_API_KEY environment variable not set."

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Analyze this portfolio and provide a brief executive summary covering:
1. Overall risk profile (VaR interpretation)
2. Top 3 risk contributors
3. Key factor exposures
4. Regime sensitivity

Keep it concise (under 300 words).

Portfolio Data:
{report.report}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text.strip()
