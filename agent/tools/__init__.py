"""Agent tools for portfolio analysis.

This package contains all tools available to the Portfolio Analysis Agent.
Tools are automatically registered when this module is imported.

Tool Categories:
    - analytics: Risk calculations (VaR, factor analysis, etc.)
    - market_data: External data fetching (prices, news, etc.)
    - portfolio: Portfolio management and information
    - charts: Chart generation for reports
"""

# Import all tool modules to trigger registration
from agent.tools import analytics
from agent.tools import market_data
from agent.tools import portfolio_tools
from agent.tools import chart_tools

__all__ = ["analytics", "market_data", "portfolio_tools", "chart_tools"]
