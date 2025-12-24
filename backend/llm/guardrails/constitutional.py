"""Constitutional AI guardrails for self-critique and grounding verification.

This module implements the Constitutional AI pattern where the model
evaluates its own responses for accuracy, appropriateness, and grounding
in the provided source data.
"""

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import anthropic


@dataclass
class GuardResult:
    """Result of a guardrail check (imported to avoid circular imports)."""
    passed: bool
    guard_name: str
    message: str
    severity: str
    details: Optional[dict] = None


@dataclass
class CritiqueResult:
    """Result of self-critique analysis."""
    factual_accuracy: float  # 0-1 score
    appropriate_uncertainty: bool
    financial_responsibility: bool
    completeness: float  # 0-1 score
    issues: List[str]
    suggestions: List[str]
    overall_score: float
    should_revise: bool


@dataclass
class GroundingResult:
    """Result of grounding verification."""
    grounded: bool
    confidence: float
    ungrounded_claims: List[str]
    supported_claims: List[str]
    explanation: str


class ConstitutionalGuard:
    """Implements Constitutional AI self-critique patterns.

    Uses a separate LLM call to evaluate responses for:
    - Factual accuracy against source data
    - Appropriate expression of uncertainty
    - Financial responsibility (no specific advice)
    - Completeness of answers
    - Grounding in provided data
    """

    CRITIQUE_PROMPT = """You are a rigorous fact-checker and quality reviewer for financial analysis.

Review the following AI response and evaluate it on these criteria:

1. **Factual Accuracy** (0-1): Are numerical claims and statements supported by the source data?
2. **Appropriate Uncertainty**: Does it acknowledge limitations and uncertainties appropriately?
3. **Financial Responsibility**: Does it avoid giving specific investment advice?
4. **Completeness** (0-1): Does it fully address the user's question?

User's Question:
{question}

AI's Response:
{response}

Source Data Available:
{source_data}

Provide your evaluation as JSON with this exact structure:
{{
    "factual_accuracy": <0.0-1.0>,
    "appropriate_uncertainty": <true/false>,
    "financial_responsibility": <true/false>,
    "completeness": <0.0-1.0>,
    "issues": ["list of specific issues found"],
    "suggestions": ["list of suggested improvements"],
    "overall_score": <0.0-1.0>,
    "should_revise": <true/false>
}}

Return ONLY the JSON, no other text."""

    GROUNDING_PROMPT = """You are a grounding verification system that checks if claims are supported by source data.

Analyze the following response and identify:
1. Claims that ARE supported by the source data
2. Claims that are NOT supported (potentially hallucinated)

Response to analyze:
{response}

Source Data:
{source_data}

Provide your analysis as JSON:
{{
    "grounded": <true if all major claims are supported>,
    "confidence": <0.0-1.0 confidence in the grounding assessment>,
    "supported_claims": ["list of claims that are supported by source data"],
    "ungrounded_claims": ["list of claims that lack support in source data"],
    "explanation": "Brief explanation of the grounding assessment"
}}

Return ONLY the JSON, no other text."""

    REVISION_PROMPT = """You are revising an AI response to address identified issues.

Original Response:
{response}

Issues to Address:
{issues}

Suggestions:
{suggestions}

Source Data (for accuracy):
{source_data}

Provide a revised response that:
1. Fixes the identified factual issues
2. Adds appropriate uncertainty language where needed
3. Ensures no specific investment advice is given
4. Maintains helpfulness while being more accurate

Revised Response:"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """Initialize constitutional guard.

        Args:
            api_key: Anthropic API key (uses env var if not provided).
            model: Model to use for critique (should be fast/cheap).
        """
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def critique_response(
        self,
        question: str,
        response: str,
        source_data: str
    ) -> CritiqueResult:
        """Have the model critique its own response.

        Args:
            question: Original user question.
            response: AI response to critique.
            source_data: Source data the response should be grounded in.

        Returns:
            CritiqueResult with detailed evaluation.
        """
        prompt = self.CRITIQUE_PROMPT.format(
            question=question[:1000],
            response=response[:2000],
            source_data=source_data[:3000]
        )

        try:
            api_response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = api_response.content[0].text.strip()

            # Parse JSON response
            result_json = self._parse_json(result_text)

            return CritiqueResult(
                factual_accuracy=result_json.get("factual_accuracy", 0.5),
                appropriate_uncertainty=result_json.get("appropriate_uncertainty", True),
                financial_responsibility=result_json.get("financial_responsibility", True),
                completeness=result_json.get("completeness", 0.5),
                issues=result_json.get("issues", []),
                suggestions=result_json.get("suggestions", []),
                overall_score=result_json.get("overall_score", 0.5),
                should_revise=result_json.get("should_revise", False)
            )

        except Exception as e:
            # Return safe defaults on error
            return CritiqueResult(
                factual_accuracy=0.5,
                appropriate_uncertainty=True,
                financial_responsibility=True,
                completeness=0.5,
                issues=[f"Critique failed: {str(e)}"],
                suggestions=[],
                overall_score=0.5,
                should_revise=False
            )

    def check_grounding(
        self,
        response: str,
        source_data: str
    ) -> GroundingResult:
        """Verify response is grounded in source data.

        Args:
            response: AI response to check.
            source_data: Source data claims should be grounded in.

        Returns:
            GroundingResult with grounding assessment.
        """
        prompt = self.GROUNDING_PROMPT.format(
            response=response[:2000],
            source_data=source_data[:3000]
        )

        try:
            api_response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = api_response.content[0].text.strip()
            result_json = self._parse_json(result_text)

            return GroundingResult(
                grounded=result_json.get("grounded", True),
                confidence=result_json.get("confidence", 0.5),
                ungrounded_claims=result_json.get("ungrounded_claims", []),
                supported_claims=result_json.get("supported_claims", []),
                explanation=result_json.get("explanation", "")
            )

        except Exception as e:
            return GroundingResult(
                grounded=True,
                confidence=0.5,
                ungrounded_claims=[],
                supported_claims=[],
                explanation=f"Grounding check failed: {str(e)}"
            )

    def revise_if_needed(
        self,
        response: str,
        critique: CritiqueResult,
        source_data: str
    ) -> str:
        """Revise response based on critique if needed.

        Args:
            response: Original response.
            critique: Critique results.
            source_data: Source data for accuracy.

        Returns:
            Revised response (or original if no revision needed).
        """
        if not critique.should_revise:
            return response

        if not critique.issues:
            return response

        prompt = self.REVISION_PROMPT.format(
            response=response[:2000],
            issues="\n".join(f"- {issue}" for issue in critique.issues),
            suggestions="\n".join(f"- {s}" for s in critique.suggestions),
            source_data=source_data[:3000]
        )

        try:
            api_response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            return api_response.content[0].text.strip()

        except Exception:
            return response  # Return original on error

    def run_full_evaluation(
        self,
        question: str,
        response: str,
        source_data: str,
        auto_revise: bool = False
    ) -> tuple:
        """Run complete constitutional evaluation.

        Args:
            question: Original user question.
            response: AI response to evaluate.
            source_data: Source data for grounding.
            auto_revise: Whether to auto-revise problematic responses.

        Returns:
            Tuple of (final_response, list of GuardResults, critique, grounding).
        """
        results = []
        final_response = response

        # Run critique
        critique = self.critique_response(question, response, source_data)

        # Convert critique to GuardResult
        if critique.overall_score < 0.5:
            results.append(GuardResult(
                passed=False,
                guard_name="constitutional_critique",
                message=f"Low quality score: {critique.overall_score:.0%}",
                severity="warning",
                details={
                    "score": critique.overall_score,
                    "issues": critique.issues,
                    "factual_accuracy": critique.factual_accuracy
                }
            ))
        else:
            results.append(GuardResult(
                passed=True,
                guard_name="constitutional_critique",
                message=f"Quality score: {critique.overall_score:.0%}",
                severity="info",
                details={"score": critique.overall_score}
            ))

        # Check financial responsibility
        if not critique.financial_responsibility:
            results.append(GuardResult(
                passed=False,
                guard_name="financial_responsibility",
                message="Response may contain specific investment advice",
                severity="warning"
            ))

        # Check uncertainty
        if not critique.appropriate_uncertainty:
            results.append(GuardResult(
                passed=True,
                guard_name="uncertainty_check",
                message="Response could express more uncertainty",
                severity="info"
            ))

        # Run grounding check
        grounding = self.check_grounding(response, source_data)

        if not grounding.grounded:
            results.append(GuardResult(
                passed=False,
                guard_name="grounding_check",
                message=f"Found {len(grounding.ungrounded_claims)} ungrounded claims",
                severity="warning",
                details={
                    "ungrounded": grounding.ungrounded_claims,
                    "confidence": grounding.confidence
                }
            ))
        else:
            results.append(GuardResult(
                passed=True,
                guard_name="grounding_check",
                message=f"Response grounded ({grounding.confidence:.0%} confidence)",
                severity="info"
            ))

        # Auto-revise if requested and needed
        if auto_revise and critique.should_revise:
            final_response = self.revise_if_needed(response, critique, source_data)
            results.append(GuardResult(
                passed=True,
                guard_name="auto_revision",
                message="Response was revised to address issues",
                severity="info",
                details={"issues_addressed": critique.issues}
            ))

        return final_response, results, critique, grounding

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks."""
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())
