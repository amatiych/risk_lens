"""Input guardrails for validating and sanitizing user inputs.

This module provides guards against prompt injection attacks, PII exposure,
and validates input data formats before sending to the LLM.
"""

import re
import base64
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GuardResult:
    """Result of a guardrail check (imported to avoid circular imports)."""
    passed: bool
    guard_name: str
    message: str
    severity: str
    details: Optional[dict] = None


class InputGuardrails:
    """Guards for validating and sanitizing user inputs.

    Provides protection against:
    - Prompt injection attacks
    - PII exposure
    - Invalid input formats
    - Malicious encoded payloads
    """

    # Patterns that suggest prompt injection attempts
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "instruction_override"),
        (r"disregard\s+(all\s+)?(previous|prior|above)", "instruction_override"),
        (r"forget\s+(everything|all|your)\s+(previous|prior|you)", "instruction_override"),

        # Role manipulation
        (r"you\s+are\s+now\s+(?!analyzing|reviewing|looking)", "role_manipulation"),
        (r"act\s+as\s+(if\s+you\s+are|a)", "role_manipulation"),
        (r"pretend\s+(to\s+be|you\s+are)", "role_manipulation"),
        (r"roleplay\s+as", "role_manipulation"),

        # System prompt extraction
        (r"(show|reveal|print|display|repeat)\s+(your\s+)?(system\s+prompt|instructions)", "system_extraction"),
        (r"what\s+(are|were)\s+your\s+(original\s+)?instructions", "system_extraction"),

        # Jailbreak attempts
        (r"do\s+anything\s+now", "jailbreak"),
        (r"(dan|developer)\s+mode", "jailbreak"),
        (r"bypass\s+(your\s+)?(restrictions|limitations|filters)", "jailbreak"),

        # Command injection in prompts
        (r"\[\s*SYSTEM\s*\]", "fake_system"),
        (r"\[\s*INST\s*\]", "fake_system"),
        (r"<\s*\|.*\|\s*>", "special_tokens"),
    ]

    # PII patterns
    PII_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "ssn", "Social Security Number detected"),
        (r"\b\d{16}\b", "credit_card", "Possible credit card number detected"),
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit_card", "Possible credit card number detected"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email", "Email address detected"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone", "Phone number detected"),
    ]

    # Valid ticker pattern
    TICKER_PATTERN = r"^[A-Z]{1,5}$"

    def __init__(self):
        """Initialize the input guardrails."""
        self._compiled_injection = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.INJECTION_PATTERNS
        ]
        self._compiled_pii = [
            (re.compile(p), name, msg)
            for p, name, msg in self.PII_PATTERNS
        ]

    def check_prompt_injection(self, text: str) -> GuardResult:
        """Check for prompt injection attempts.

        Args:
            text: User input text to check.

        Returns:
            GuardResult indicating if injection was detected.
        """
        text_lower = text.lower()
        detected = []

        for pattern, attack_type in self._compiled_injection:
            if pattern.search(text):
                detected.append(attack_type)

        # Check for base64 encoded payloads
        if self._contains_base64_payload(text):
            detected.append("encoded_payload")

        if detected:
            return GuardResult(
                passed=False,
                guard_name="prompt_injection",
                message=f"Potential prompt injection detected: {', '.join(set(detected))}",
                severity="block",
                details={"attack_types": list(set(detected))}
            )

        return GuardResult(
            passed=True,
            guard_name="prompt_injection",
            message="No prompt injection detected",
            severity="info"
        )

    def _contains_base64_payload(self, text: str) -> bool:
        """Check if text contains suspicious base64 encoded content."""
        # Look for base64-like strings (at least 20 chars, typical base64 charset)
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        matches = b64_pattern.findall(text)

        for match in matches:
            try:
                # Try to decode and check if it contains suspicious content
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                # Check if decoded content has injection patterns
                for pattern, _ in self._compiled_injection:
                    if pattern.search(decoded):
                        return True
            except Exception:
                pass

        return False

    def detect_pii(self, text: str) -> GuardResult:
        """Detect personally identifiable information in input.

        Args:
            text: User input text to check.

        Returns:
            GuardResult with warning if PII detected.
        """
        detected = []

        for pattern, pii_type, message in self._compiled_pii:
            if pattern.search(text):
                detected.append({"type": pii_type, "message": message})

        if detected:
            return GuardResult(
                passed=True,  # Warning, not blocking
                guard_name="pii_detection",
                message=f"PII detected in input: {', '.join(d['type'] for d in detected)}",
                severity="warning",
                details={"pii_found": detected}
            )

        return GuardResult(
            passed=True,
            guard_name="pii_detection",
            message="No PII detected",
            severity="info"
        )

    def validate_ticker(self, ticker: str) -> GuardResult:
        """Validate a stock ticker symbol.

        Args:
            ticker: Ticker symbol to validate.

        Returns:
            GuardResult indicating if ticker is valid.
        """
        ticker = ticker.strip().upper()

        if not ticker:
            return GuardResult(
                passed=False,
                guard_name="ticker_validation",
                message="Empty ticker symbol",
                severity="error"
            )

        if not re.match(self.TICKER_PATTERN, ticker):
            return GuardResult(
                passed=False,
                guard_name="ticker_validation",
                message=f"Invalid ticker format: {ticker}",
                severity="error",
                details={"ticker": ticker, "expected_pattern": "1-5 uppercase letters"}
            )

        return GuardResult(
            passed=True,
            guard_name="ticker_validation",
            message=f"Valid ticker: {ticker}",
            severity="info"
        )

    def validate_tickers(self, tickers: List[str]) -> GuardResult:
        """Validate a list of ticker symbols.

        Args:
            tickers: List of ticker symbols to validate.

        Returns:
            GuardResult with details on invalid tickers.
        """
        invalid = []
        for ticker in tickers:
            result = self.validate_ticker(ticker)
            if not result.passed:
                invalid.append(ticker)

        if invalid:
            return GuardResult(
                passed=False,
                guard_name="tickers_validation",
                message=f"Invalid tickers: {', '.join(invalid)}",
                severity="error",
                details={"invalid_tickers": invalid, "valid_count": len(tickers) - len(invalid)}
            )

        return GuardResult(
            passed=True,
            guard_name="tickers_validation",
            message=f"All {len(tickers)} tickers valid",
            severity="info"
        )

    def validate_numeric_range(
        self,
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        field_name: str = "value"
    ) -> GuardResult:
        """Validate a numeric value is within expected range.

        Args:
            value: Numeric value to validate.
            min_val: Minimum allowed value (inclusive).
            max_val: Maximum allowed value (inclusive).
            field_name: Name of the field for error messages.

        Returns:
            GuardResult indicating if value is valid.
        """
        if min_val is not None and value < min_val:
            return GuardResult(
                passed=False,
                guard_name="numeric_range",
                message=f"{field_name} ({value}) below minimum ({min_val})",
                severity="error",
                details={"value": value, "min": min_val, "field": field_name}
            )

        if max_val is not None and value > max_val:
            return GuardResult(
                passed=False,
                guard_name="numeric_range",
                message=f"{field_name} ({value}) above maximum ({max_val})",
                severity="error",
                details={"value": value, "max": max_val, "field": field_name}
            )

        return GuardResult(
            passed=True,
            guard_name="numeric_range",
            message=f"{field_name} within valid range",
            severity="info"
        )

    def sanitize_input(self, text: str) -> Tuple[str, List[str]]:
        """Sanitize user input by removing potentially harmful content.

        Args:
            text: User input to sanitize.

        Returns:
            Tuple of (sanitized text, list of modifications made).
        """
        modifications = []
        sanitized = text

        # Remove null bytes
        if '\x00' in sanitized:
            sanitized = sanitized.replace('\x00', '')
            modifications.append("removed_null_bytes")

        # Remove control characters (except newlines and tabs)
        control_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        if control_chars.search(sanitized):
            sanitized = control_chars.sub('', sanitized)
            modifications.append("removed_control_chars")

        # Limit length (prevent token exhaustion)
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            modifications.append(f"truncated_to_{max_length}_chars")

        return sanitized, modifications

    def run_all_checks(self, text: str) -> List[GuardResult]:
        """Run all input guardrails on the given text.

        Args:
            text: User input to check.

        Returns:
            List of all GuardResult objects from checks.
        """
        results = []

        # Sanitize first
        sanitized, mods = self.sanitize_input(text)
        if mods:
            results.append(GuardResult(
                passed=True,
                guard_name="input_sanitization",
                message=f"Input sanitized: {', '.join(mods)}",
                severity="info",
                details={"modifications": mods}
            ))

        # Check for injection
        results.append(self.check_prompt_injection(sanitized))

        # Check for PII
        results.append(self.detect_pii(sanitized))

        return results
