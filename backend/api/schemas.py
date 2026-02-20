"""Pydantic request/response models for the Risk Lens API."""

from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional


class HoldingResponse(BaseModel):
    ticker: str
    shares: float
    price: Optional[float] = None
    market_value: Optional[float] = None
    weight: Optional[float] = None


class PortfolioResponse(BaseModel):
    id: str
    name: str
    nav: float
    holdings: List[HoldingResponse]
    status: Literal["uploaded", "enriched", "analyzed"] = "uploaded"


class VarResultResponse(BaseModel):
    ci: float
    var: float
    es: float
    var_date: str
    marginal_var: List[Dict[str, Any]]
    incremental_var: List[Dict[str, Any]]


class FactorExposureResponse(BaseModel):
    factor: str
    beta: float
    risk_contribution: float


class RegimeStatResponse(BaseModel):
    regime: int
    label: str
    description: str
    performance: float


class PcaResponse(BaseModel):
    variance_explained: List[float]
    cumulative_variance: List[float]


class CorrelationResponse(BaseModel):
    tickers: List[str]
    matrix: List[List[float]]


class AnalysisResponse(BaseModel):
    portfolio: PortfolioResponse
    var_results: List[VarResultResponse]
    correlation: CorrelationResponse
    factor_exposures: List[FactorExposureResponse]
    regime_stats: List[RegimeStatResponse]
    pca: PcaResponse
    ai_summary: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class ProviderResponse(BaseModel):
    provider: Literal["claude", "openai"]


class ProviderUpdateRequest(BaseModel):
    provider: Literal["claude", "openai"]


class ErrorResponse(BaseModel):
    detail: str
