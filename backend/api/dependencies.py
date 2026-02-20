"""In-memory store for portfolio and analysis results."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from frontend.services.portfolio_service import AnalysisResults
from models.portfolio import Portfolio
from backend.reporting.portfolio_report import PortfolioReport


@dataclass
class PortfolioEntry:
    portfolio: Portfolio
    results: Optional[AnalysisResults] = None
    report: Optional[PortfolioReport] = None
    ai_summary: Optional[str] = None
    chat_history: list = field(default_factory=list)


class PortfolioStore:
    """Simple in-memory store keyed by UUID string."""

    def __init__(self):
        self._data: Dict[str, PortfolioEntry] = {}

    def set(self, portfolio_id: str, entry: PortfolioEntry):
        self._data[portfolio_id] = entry

    def get(self, portfolio_id: str) -> Optional[PortfolioEntry]:
        return self._data.get(portfolio_id)

    def exists(self, portfolio_id: str) -> bool:
        return portfolio_id in self._data


store = PortfolioStore()

# Global LLM provider setting
_llm_provider: str = "claude"


def get_provider() -> str:
    return _llm_provider


def set_provider(provider: str):
    global _llm_provider
    _llm_provider = provider
