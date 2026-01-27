"""Portfolio Analysis Agent - AI-powered portfolio risk analysis.

This package provides an AI agent that analyzes investment portfolios
using a flexible tool-based architecture. The agent can load portfolios,
run various risk analytics, and generate comprehensive reports with charts.
"""

from agent.core import PortfolioAgent
from agent.tool_registry import ToolRegistry, tool

__all__ = ["PortfolioAgent", "ToolRegistry", "tool"]
