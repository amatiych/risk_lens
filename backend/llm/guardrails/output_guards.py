"""Output guardrails for validating and enhancing LLM responses.

This module provides guards for detecting hallucinations, adding required
disclaimers, calculating response confidence, and validating numerical claims.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GuardResult:
    """Result of a guardrail check (imported to avoid circular imports)."""
    passed: bool
    guard_name: str
    message: str
    severity: str
    details: Optional[dict] = None


@dataclass
class NumericalClaim:
    """A numerical claim extracted from response text."""
    value: float
    context: str
    original_text: str
    unit: Optional[str] = None


@dataclass
class HallucinationViolation:
    """A detected hallucination in the response."""
    claim: str
    claimed_value: float
    actual_value: Optional[float]
    discrepancy: Optional[float]
    explanation: str


class OutputGuardrails:
    """Guards for validating and enhancing LLM responses.

    Provides:
    - Hallucination detection by cross-checking with source data
    - Automatic financial disclaimer injection
    - Response confidence scoring
    - Refusal detection
    - Numerical claim validation
    """

    FINANCIAL_DISCLAIMER = (
        "\n\n---\n"
        "*Disclaimer: This analysis is for informational purposes only and does not "
        "constitute financial advice. Past performance does not guarantee future results. "
        "Please consult a qualified financial advisor before making investment decisions.*"
    )

    # Patterns suggesting the model refused to answer
    REFUSAL_PATTERNS = [
        r"I (?:cannot|can't|won't|will not|am not able to)",
        r"I'm (?:sorry|afraid|unable)",
        r"(?:cannot|can't) provide (?:financial|investment) advice",
        r"(?:not|never) intended (?:as|to be) (?:financial|investment) advice",
        r"consult (?:a|with|your) (?:qualified )?(?:financial )?(?:advisor|professional)",
    ]

    # Patterns for extracting numerical claims
    NUMERICAL_PATTERNS = [
        # Percentages
        (r'(\d+(?:\.\d+)?)\s*%', 'percentage'),
        # Dollar amounts
        (r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?', 'currency'),
        # VaR values
        (r'VaR\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*%?', 'var'),
        # Generic numbers with context
        (r'(\d+(?:\.\d+)?)\s*(?:basis points|bps)', 'basis_points'),
    ]

    # Keywords suggesting uncertainty
    UNCERTAINTY_KEYWORDS = [
        "approximately", "about", "around", "roughly", "estimated",
        "may", "might", "could", "possibly", "potentially",
        "uncertain", "unclear", "unknown", "not sure", "difficult to determine"
    ]

    def __init__(self):
        """Initialize the output guardrails."""
        self._compiled_refusal = [
            re.compile(p, re.IGNORECASE) for p in self.REFUSAL_PATTERNS
        ]
        self._compiled_numerical = [
            (re.compile(p, re.IGNORECASE), unit_type)
            for p, unit_type in self.NUMERICAL_PATTERNS
        ]

    def check_hallucination(
        self,
        response: str,
        source_data: Dict[str, Any],
        tolerance: float = 0.1
    ) -> GuardResult:
        """Check for potential hallucinations by comparing claims to source data.

        Args:
            response: The LLM response text.
            source_data: Dictionary of actual data to check against.
            tolerance: Allowed deviation as fraction (0.1 = 10%).

        Returns:
            GuardResult with details on any detected hallucinations.
        """
        violations = self.validate_numerical_claims(response, source_data, tolerance)

        if violations:
            return GuardResult(
                passed=False,
                guard_name="hallucination_check",
                message=f"Detected {len(violations)} potential hallucination(s)",
                severity="warning",
                details={
                    "violations": [
                        {
                            "claim": v.claim,
                            "claimed": v.claimed_value,
                            "actual": v.actual_value,
                            "discrepancy": v.discrepancy,
                            "explanation": v.explanation
                        }
                        for v in violations
                    ]
                }
            )

        return GuardResult(
            passed=True,
            guard_name="hallucination_check",
            message="No hallucinations detected",
            severity="info"
        )

    def validate_numerical_claims(
        self,
        response: str,
        source_data: Dict[str, Any],
        tolerance: float = 0.1
    ) -> List[HallucinationViolation]:
        """Extract and validate numerical claims against source data.

        Args:
            response: The LLM response text.
            source_data: Dictionary of actual data to check against.
            tolerance: Allowed deviation as fraction.

        Returns:
            List of HallucinationViolation for claims that don't match source.
        """
        violations = []
        claims = self._extract_numerical_claims(response)

        # Flatten source data for easier lookup
        flat_data = self._flatten_dict(source_data)

        for claim in claims:
            # Try to find matching value in source data
            match = self._find_matching_source(claim, flat_data, tolerance)
            if match:
                actual_value, discrepancy = match
                if discrepancy is not None and abs(discrepancy) > tolerance:
                    violations.append(HallucinationViolation(
                        claim=claim.original_text,
                        claimed_value=claim.value,
                        actual_value=actual_value,
                        discrepancy=discrepancy,
                        explanation=f"Claimed {claim.value} but actual is {actual_value} ({discrepancy:.1%} difference)"
                    ))

        return violations

    def _extract_numerical_claims(self, text: str) -> List[NumericalClaim]:
        """Extract numerical claims from text."""
        claims = []

        for pattern, unit_type in self._compiled_numerical:
            for match in pattern.finditer(text):
                try:
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]

                    # Parse the value
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)

                    claims.append(NumericalClaim(
                        value=value,
                        context=context,
                        original_text=match.group(0),
                        unit=unit_type
                    ))
                except (ValueError, IndexError):
                    continue

        return claims

    def _flatten_dict(self, d: Dict, prefix: str = '') -> Dict[str, float]:
        """Flatten a nested dictionary to key-value pairs."""
        items = {}
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            elif isinstance(v, (int, float)):
                items[new_key] = float(v)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}[{i}]"))
                    elif isinstance(item, (int, float)):
                        items[f"{new_key}[{i}]"] = float(item)
        return items

    def _find_matching_source(
        self,
        claim: NumericalClaim,
        flat_data: Dict[str, float],
        tolerance: float
    ) -> Optional[Tuple[float, float]]:
        """Find source data that might match a claim."""
        # Look for values close to the claimed value
        for key, actual in flat_data.items():
            if actual == 0:
                continue

            discrepancy = (claim.value - actual) / actual

            # Check if this could be the same value
            if abs(discrepancy) < 0.5:  # Within 50% - might be the same metric
                return (actual, discrepancy)

        return None

    def add_disclaimer(self, response: str) -> str:
        """Add financial disclaimer to response if not already present.

        Args:
            response: The LLM response text.

        Returns:
            Response with disclaimer appended if needed.
        """
        # Check if disclaimer-like content already exists
        disclaimer_indicators = [
            "not constitute financial advice",
            "informational purposes only",
            "consult a qualified",
            "does not guarantee"
        ]

        for indicator in disclaimer_indicators:
            if indicator.lower() in response.lower():
                return response

        return response + self.FINANCIAL_DISCLAIMER

    def check_disclaimer_present(self, response: str) -> GuardResult:
        """Check if response contains appropriate disclaimers.

        Args:
            response: The LLM response text.

        Returns:
            GuardResult indicating disclaimer status.
        """
        disclaimer_indicators = [
            "not financial advice",
            "informational purposes",
            "consult a qualified",
            "past performance"
        ]

        found = any(
            indicator.lower() in response.lower()
            for indicator in disclaimer_indicators
        )

        if found:
            return GuardResult(
                passed=True,
                guard_name="disclaimer_check",
                message="Response contains appropriate disclaimers",
                severity="info"
            )

        return GuardResult(
            passed=True,  # Warning, not failure
            guard_name="disclaimer_check",
            message="Response may need disclaimer added",
            severity="warning",
            details={"action": "disclaimer_will_be_appended"}
        )

    def calculate_confidence(self, response: str) -> Tuple[float, GuardResult]:
        """Calculate a confidence score for the response.

        Factors considered:
        - Presence of uncertainty language
        - Specificity of claims
        - Use of hedging words

        Args:
            response: The LLM response text.

        Returns:
            Tuple of (confidence score 0-1, GuardResult with details).
        """
        response_lower = response.lower()

        # Count uncertainty indicators
        uncertainty_count = sum(
            1 for keyword in self.UNCERTAINTY_KEYWORDS
            if keyword in response_lower
        )

        # Normalize by response length
        word_count = len(response.split())
        uncertainty_ratio = uncertainty_count / max(word_count / 100, 1)

        # Base confidence starts high
        confidence = 1.0

        # Reduce for uncertainty language
        confidence -= min(uncertainty_ratio * 0.2, 0.3)

        # Reduce for very short responses (might be refusals)
        if word_count < 50:
            confidence -= 0.1

        # Reduce for lack of specific data
        has_numbers = bool(re.search(r'\d+(?:\.\d+)?', response))
        if not has_numbers:
            confidence -= 0.1

        confidence = max(0.0, min(1.0, confidence))

        severity = "info"
        if confidence < 0.5:
            severity = "warning"
        elif confidence < 0.3:
            severity = "error"

        return confidence, GuardResult(
            passed=confidence >= 0.3,
            guard_name="confidence_score",
            message=f"Response confidence: {confidence:.0%}",
            severity=severity,
            details={
                "confidence": confidence,
                "uncertainty_words": uncertainty_count,
                "word_count": word_count
            }
        )

    def check_refusal(self, response: str) -> GuardResult:
        """Check if the model refused to answer.

        Args:
            response: The LLM response text.

        Returns:
            GuardResult indicating if refusal detected.
        """
        for pattern in self._compiled_refusal:
            if pattern.search(response):
                return GuardResult(
                    passed=True,  # Not a failure, just informational
                    guard_name="refusal_detection",
                    message="Model declined to fully answer",
                    severity="warning",
                    details={"pattern_matched": pattern.pattern}
                )

        return GuardResult(
            passed=True,
            guard_name="refusal_detection",
            message="No refusal detected",
            severity="info"
        )

    def check_response_length(
        self,
        response: str,
        min_length: int = 50,
        max_length: int = 10000
    ) -> GuardResult:
        """Check if response length is within acceptable bounds.

        Args:
            response: The LLM response text.
            min_length: Minimum acceptable character length.
            max_length: Maximum acceptable character length.

        Returns:
            GuardResult indicating length status.
        """
        length = len(response)

        if length < min_length:
            return GuardResult(
                passed=False,
                guard_name="response_length",
                message=f"Response too short ({length} chars, min {min_length})",
                severity="warning",
                details={"length": length, "min": min_length}
            )

        if length > max_length:
            return GuardResult(
                passed=True,  # Just a warning
                guard_name="response_length",
                message=f"Response very long ({length} chars)",
                severity="info",
                details={"length": length, "max": max_length}
            )

        return GuardResult(
            passed=True,
            guard_name="response_length",
            message=f"Response length acceptable ({length} chars)",
            severity="info"
        )

    def run_all_checks(
        self,
        response: str,
        source_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[GuardResult]]:
        """Run all output guardrails on the response.

        Args:
            response: The LLM response text.
            source_data: Optional source data for hallucination checking.

        Returns:
            Tuple of (processed response with disclaimer, list of GuardResults).
        """
        results = []

        # Check for refusal
        results.append(self.check_refusal(response))

        # Check response length
        results.append(self.check_response_length(response))

        # Calculate confidence
        confidence, conf_result = self.calculate_confidence(response)
        results.append(conf_result)

        # Check for hallucinations if source data provided
        if source_data:
            results.append(self.check_hallucination(response, source_data))

        # Check and add disclaimer
        results.append(self.check_disclaimer_present(response))
        processed_response = self.add_disclaimer(response)

        return processed_response, results
