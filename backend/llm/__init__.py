"""LLM integration module for AI-powered risk analysis.

This module provides LLM integration (Claude, OpenAI) for portfolio risk analysis,
including tool definitions for runtime data fetching and structured
output schemas for consistent responses.

Example:
    # Default (Claude)
    from backend.llm import RiskAnalyzer
    analyzer = RiskAnalyzer(report)

    # With OpenAI
    from backend.llm import RiskAnalyzer, LLMConfig
    config = LLMConfig(provider="openai")
    analyzer = RiskAnalyzer(report, config=config)
"""

from backend.llm.risk_analyzer import RiskAnalyzer
from backend.llm.tools import TOOLS, execute_tool
from backend.llm.schemas import VAR_ANALYSIS_SCHEMA, VaRAnalysisResult
from backend.llm.providers import LLMConfig, get_provider

__all__ = [
    "RiskAnalyzer",
    "TOOLS",
    "execute_tool",
    "VAR_ANALYSIS_SCHEMA",
    "VaRAnalysisResult",
    "LLMConfig",
    "get_provider",
]
