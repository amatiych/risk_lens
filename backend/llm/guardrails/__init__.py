"""Guardrails module for AI safety and responsible LLM usage.

This module provides comprehensive guardrails for the Risk Lens application,
including input validation, output verification, process controls, and
constitutional AI patterns for self-critique.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
import json


@dataclass
class GuardResult:
    """Result of a single guardrail check.

    Attributes:
        passed: Whether the check passed without issues.
        guard_name: Name of the guardrail that ran.
        message: Human-readable description of the result.
        severity: Level of concern (info, warning, error, block).
        details: Optional additional context or data.
    """
    passed: bool
    guard_name: str
    message: str
    severity: Literal["info", "warning", "error", "block"]
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "guard_name": self.guard_name,
            "message": self.message,
            "severity": self.severity,
            "details": self.details
        }


@dataclass
class GuardrailsReport:
    """Complete report of all guardrail checks for a request/response cycle.

    Attributes:
        input_checks: Results from input validation guardrails.
        output_checks: Results from output validation guardrails.
        constitutional_checks: Results from self-critique guardrails.
        process_checks: Results from process guardrails (rate limiting, etc.).
        overall_passed: True if no blocking issues found.
        blocked: True if request was blocked by a guardrail.
        timestamp: When the checks were run.
    """
    input_checks: List[GuardResult] = field(default_factory=list)
    output_checks: List[GuardResult] = field(default_factory=list)
    constitutional_checks: List[GuardResult] = field(default_factory=list)
    process_checks: List[GuardResult] = field(default_factory=list)
    overall_passed: bool = True
    blocked: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def add_input_check(self, result: GuardResult) -> None:
        """Add an input guardrail result."""
        self.input_checks.append(result)
        self._update_status(result)

    def add_output_check(self, result: GuardResult) -> None:
        """Add an output guardrail result."""
        self.output_checks.append(result)
        self._update_status(result)

    def add_constitutional_check(self, result: GuardResult) -> None:
        """Add a constitutional guardrail result."""
        self.constitutional_checks.append(result)
        self._update_status(result)

    def add_process_check(self, result: GuardResult) -> None:
        """Add a process guardrail result."""
        self.process_checks.append(result)
        self._update_status(result)

    def _update_status(self, result: GuardResult) -> None:
        """Update overall status based on a new result."""
        if result.severity == "block":
            self.blocked = True
            self.overall_passed = False
        elif result.severity == "error" and not result.passed:
            self.overall_passed = False

    def get_warnings(self) -> List[GuardResult]:
        """Get all warning-level results."""
        all_checks = (
            self.input_checks +
            self.output_checks +
            self.constitutional_checks +
            self.process_checks
        )
        return [r for r in all_checks if r.severity == "warning"]

    def get_errors(self) -> List[GuardResult]:
        """Get all error-level results."""
        all_checks = (
            self.input_checks +
            self.output_checks +
            self.constitutional_checks +
            self.process_checks
        )
        return [r for r in all_checks if r.severity in ("error", "block")]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_checks": [r.to_dict() for r in self.input_checks],
            "output_checks": [r.to_dict() for r in self.output_checks],
            "constitutional_checks": [r.to_dict() for r in self.constitutional_checks],
            "process_checks": [r.to_dict() for r in self.process_checks],
            "overall_passed": self.overall_passed,
            "blocked": self.blocked,
            "timestamp": self.timestamp.isoformat()
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Get a human-readable summary of the report."""
        total = (
            len(self.input_checks) +
            len(self.output_checks) +
            len(self.constitutional_checks) +
            len(self.process_checks)
        )
        warnings = len(self.get_warnings())
        errors = len(self.get_errors())

        status = "BLOCKED" if self.blocked else ("PASSED" if self.overall_passed else "FAILED")
        return f"Guardrails: {status} ({total} checks, {warnings} warnings, {errors} errors)"


# Export all public classes
from backend.llm.guardrails.input_guards import InputGuardrails
from backend.llm.guardrails.output_guards import OutputGuardrails
from backend.llm.guardrails.process_guards import ProcessGuardrails, GuardedClient
from backend.llm.guardrails.constitutional import ConstitutionalGuard

__all__ = [
    "GuardResult",
    "GuardrailsReport",
    "InputGuardrails",
    "OutputGuardrails",
    "ProcessGuardrails",
    "GuardedClient",
    "ConstitutionalGuard",
]
