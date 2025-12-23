"""Response schemas for structured Claude AI outputs.

This module defines JSON schemas for consistent, parseable responses
from Claude AI analysis. These schemas ensure the AI returns data
in a predictable format that can be reliably processed.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json


# Schema for VaR analysis structured output
VAR_ANALYSIS_SCHEMA = {
    "name": "var_analysis_result",
    "description": "Structured VaR analysis result with risk drivers, concentration analysis, and recommendations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "Brief overview of portfolio risk profile (2-3 sentences)"
            },
            "risk_drivers": {
                "type": "array",
                "description": "List of positions ranked by risk contribution",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "company_name": {"type": "string"},
                        "sector": {"type": "string"},
                        "marginal_var_contribution": {"type": "number"},
                        "incremental_var_contribution": {"type": "number"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["ticker", "marginal_var_contribution", "explanation"],
                    "additionalProperties": False
                }
            },
            "concentration_analysis": {
                "type": "object",
                "properties": {
                    "most_concentrated_positions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers with highest concentration risk"
                    },
                    "best_diversifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Positions providing best diversification"
                    },
                    "concentration_risk_level": {
                        "type": "string",
                        "enum": ["low", "moderate", "high", "very_high"]
                    },
                    "concentration_summary": {"type": "string"}
                },
                "required": ["most_concentrated_positions", "best_diversifiers", "concentration_risk_level", "concentration_summary"],
                "additionalProperties": False
            },
            "factor_interpretation": {
                "type": "object",
                "properties": {
                    "dominant_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Factors with largest risk contribution"
                    },
                    "factor_exposures": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "factor": {"type": "string"},
                                "beta": {"type": "number"},
                                "risk_contribution_pct": {"type": "number"},
                                "interpretation": {"type": "string"}
                            },
                            "required": ["factor", "beta", "interpretation"],
                            "additionalProperties": False
                        }
                    },
                    "factor_risk_summary": {"type": "string"}
                },
                "required": ["dominant_factors", "factor_risk_summary"],
                "additionalProperties": False
            },
            "regime_interpretation": {
                "type": "object",
                "properties": {
                    "regime_sensitivity": {
                        "type": "string",
                        "enum": ["low", "moderate", "high"]
                    },
                    "best_performing_regime": {"type": "string"},
                    "worst_performing_regime": {"type": "string"},
                    "regime_risk_summary": {"type": "string"}
                },
                "required": ["regime_sensitivity", "regime_risk_summary"],
                "additionalProperties": False
            },
            "var_date_context": {
                "type": "object",
                "properties": {
                    "var_date": {"type": "string"},
                    "market_events": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Market events that may explain VaR loss"
                    },
                    "price_movements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                                "daily_return_pct": {"type": "number"}
                            },
                            "required": ["ticker", "daily_return_pct"],
                            "additionalProperties": False
                        }
                    },
                    "context_explanation": {"type": "string"}
                },
                "required": ["var_date", "context_explanation"],
                "additionalProperties": False
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Actionable recommendations to improve risk profile"
            }
        },
        "required": [
            "executive_summary",
            "risk_drivers",
            "concentration_analysis",
            "factor_interpretation",
            "regime_interpretation",
            "recommendations"
        ],
        "additionalProperties": False
    }
}


# Schema for chat response (optional structured format)
CHAT_RESPONSE_SCHEMA = {
    "name": "chat_response",
    "description": "Structured chat response about portfolio analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Direct answer to the user's question"
            },
            "supporting_data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "value": {"type": "string"},
                        "context": {"type": "string"}
                    },
                    "required": ["metric", "value"],
                    "additionalProperties": False
                },
                "description": "Key data points supporting the answer"
            },
            "follow_up_suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Suggested follow-up questions"
            }
        },
        "required": ["answer"],
        "additionalProperties": False
    }
}


@dataclass
class RiskDriver:
    """Individual position risk contribution."""
    ticker: str
    marginal_var_contribution: float
    incremental_var_contribution: float
    explanation: str
    company_name: Optional[str] = None
    sector: Optional[str] = None


@dataclass
class ConcentrationAnalysis:
    """Portfolio concentration analysis results."""
    most_concentrated_positions: List[str]
    best_diversifiers: List[str]
    concentration_risk_level: str
    concentration_summary: str


@dataclass
class FactorExposure:
    """Individual factor exposure details."""
    factor: str
    beta: float
    interpretation: str
    risk_contribution_pct: Optional[float] = None


@dataclass
class FactorInterpretation:
    """Factor analysis interpretation."""
    dominant_factors: List[str]
    factor_risk_summary: str
    factor_exposures: Optional[List[FactorExposure]] = None


@dataclass
class RegimeInterpretation:
    """Regime analysis interpretation."""
    regime_sensitivity: str
    regime_risk_summary: str
    best_performing_regime: Optional[str] = None
    worst_performing_regime: Optional[str] = None


@dataclass
class VaRDateContext:
    """Context for VaR date market events."""
    var_date: str
    context_explanation: str
    market_events: Optional[List[str]] = None
    price_movements: Optional[List[Dict[str, Any]]] = None


@dataclass
class VaRAnalysisResult:
    """Complete structured VaR analysis result."""
    executive_summary: str
    risk_drivers: List[RiskDriver]
    concentration_analysis: ConcentrationAnalysis
    factor_interpretation: FactorInterpretation
    regime_interpretation: RegimeInterpretation
    recommendations: List[str]
    var_date_context: Optional[VaRDateContext] = None

    @classmethod
    def from_json(cls, json_str: str) -> "VaRAnalysisResult":
        """Parse JSON string into VaRAnalysisResult object.

        Args:
            json_str: JSON string matching VAR_ANALYSIS_SCHEMA.

        Returns:
            VaRAnalysisResult instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VaRAnalysisResult":
        """Create VaRAnalysisResult from dictionary.

        Args:
            data: Dictionary matching VAR_ANALYSIS_SCHEMA structure.

        Returns:
            VaRAnalysisResult instance.
        """
        risk_drivers = [
            RiskDriver(
                ticker=rd["ticker"],
                marginal_var_contribution=rd.get("marginal_var_contribution", 0),
                incremental_var_contribution=rd.get("incremental_var_contribution", 0),
                explanation=rd["explanation"],
                company_name=rd.get("company_name"),
                sector=rd.get("sector")
            )
            for rd in data.get("risk_drivers", [])
        ]

        concentration = data.get("concentration_analysis", {})
        concentration_analysis = ConcentrationAnalysis(
            most_concentrated_positions=concentration.get("most_concentrated_positions", []),
            best_diversifiers=concentration.get("best_diversifiers", []),
            concentration_risk_level=concentration.get("concentration_risk_level", "unknown"),
            concentration_summary=concentration.get("concentration_summary", "")
        )

        factor = data.get("factor_interpretation", {})
        factor_exposures = None
        if "factor_exposures" in factor:
            factor_exposures = [
                FactorExposure(
                    factor=fe["factor"],
                    beta=fe.get("beta", 0),
                    interpretation=fe["interpretation"],
                    risk_contribution_pct=fe.get("risk_contribution_pct")
                )
                for fe in factor["factor_exposures"]
            ]
        factor_interpretation = FactorInterpretation(
            dominant_factors=factor.get("dominant_factors", []),
            factor_risk_summary=factor.get("factor_risk_summary", ""),
            factor_exposures=factor_exposures
        )

        regime = data.get("regime_interpretation", {})
        regime_interpretation = RegimeInterpretation(
            regime_sensitivity=regime.get("regime_sensitivity", "unknown"),
            regime_risk_summary=regime.get("regime_risk_summary", ""),
            best_performing_regime=regime.get("best_performing_regime"),
            worst_performing_regime=regime.get("worst_performing_regime")
        )

        var_context = None
        if "var_date_context" in data:
            vc = data["var_date_context"]
            var_context = VaRDateContext(
                var_date=vc.get("var_date", ""),
                context_explanation=vc.get("context_explanation", ""),
                market_events=vc.get("market_events"),
                price_movements=vc.get("price_movements")
            )

        return cls(
            executive_summary=data.get("executive_summary", ""),
            risk_drivers=risk_drivers,
            concentration_analysis=concentration_analysis,
            factor_interpretation=factor_interpretation,
            regime_interpretation=regime_interpretation,
            recommendations=data.get("recommendations", []),
            var_date_context=var_context
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation of the analysis result.
        """
        result = {
            "executive_summary": self.executive_summary,
            "risk_drivers": [
                {
                    "ticker": rd.ticker,
                    "marginal_var_contribution": rd.marginal_var_contribution,
                    "incremental_var_contribution": rd.incremental_var_contribution,
                    "explanation": rd.explanation,
                    **({"company_name": rd.company_name} if rd.company_name else {}),
                    **({"sector": rd.sector} if rd.sector else {})
                }
                for rd in self.risk_drivers
            ],
            "concentration_analysis": {
                "most_concentrated_positions": self.concentration_analysis.most_concentrated_positions,
                "best_diversifiers": self.concentration_analysis.best_diversifiers,
                "concentration_risk_level": self.concentration_analysis.concentration_risk_level,
                "concentration_summary": self.concentration_analysis.concentration_summary
            },
            "factor_interpretation": {
                "dominant_factors": self.factor_interpretation.dominant_factors,
                "factor_risk_summary": self.factor_interpretation.factor_risk_summary
            },
            "regime_interpretation": {
                "regime_sensitivity": self.regime_interpretation.regime_sensitivity,
                "regime_risk_summary": self.regime_interpretation.regime_risk_summary
            },
            "recommendations": self.recommendations
        }

        if self.factor_interpretation.factor_exposures:
            result["factor_interpretation"]["factor_exposures"] = [
                {
                    "factor": fe.factor,
                    "beta": fe.beta,
                    "interpretation": fe.interpretation,
                    **({"risk_contribution_pct": fe.risk_contribution_pct}
                       if fe.risk_contribution_pct else {})
                }
                for fe in self.factor_interpretation.factor_exposures
            ]

        if self.regime_interpretation.best_performing_regime:
            result["regime_interpretation"]["best_performing_regime"] = \
                self.regime_interpretation.best_performing_regime
        if self.regime_interpretation.worst_performing_regime:
            result["regime_interpretation"]["worst_performing_regime"] = \
                self.regime_interpretation.worst_performing_regime

        if self.var_date_context:
            result["var_date_context"] = {
                "var_date": self.var_date_context.var_date,
                "context_explanation": self.var_date_context.context_explanation
            }
            if self.var_date_context.market_events:
                result["var_date_context"]["market_events"] = self.var_date_context.market_events
            if self.var_date_context.price_movements:
                result["var_date_context"]["price_movements"] = self.var_date_context.price_movements

        return result

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=2)
