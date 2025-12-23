"""LLM integration module for AI-powered risk analysis.

This module provides Claude AI integration for portfolio risk analysis,
including tool definitions for runtime data fetching and structured
output schemas for consistent responses.
"""

from backend.llm.var_analyzer import VaRAnalyzer
from backend.llm.tools import TOOLS, execute_tool
from backend.llm.schemas import VAR_ANALYSIS_SCHEMA, VaRAnalysisResult

__all__ = [
    "VaRAnalyzer",
    "TOOLS",
    "execute_tool",
    "VAR_ANALYSIS_SCHEMA",
    "VaRAnalysisResult",
]
