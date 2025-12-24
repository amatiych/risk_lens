"""Red Team Test Runner for executing adversarial attacks.

This module provides the test runner that executes attack payloads
against the guardrails system and collects results for analysis.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from backend.llm.guardrails import GuardrailsReport
from backend.llm.guardrails.input_guards import InputGuardrails
from backend.llm.guardrails.output_guards import OutputGuardrails
from backend.llm.redteam.attacks import (
    AttackPayload,
    AttackCategory,
    ATTACK_PAYLOADS,
    get_attacks_by_category,
    get_attacks_by_severity,
)


@dataclass
class AttackResult:
    """Result of a single attack attempt.

    Attributes:
        attack: The attack payload that was tested.
        blocked: Whether the guardrails blocked the attack.
        expected_blocked: Whether the attack should have been blocked.
        correct: Whether the result matches expectation.
        detection_time_ms: Time taken to detect/process in milliseconds.
        guard_triggered: Name of the guard that triggered (if any).
        details: Additional details from the guardrails.
        error: Error message if attack execution failed.
    """
    attack: AttackPayload
    blocked: bool
    expected_blocked: bool
    correct: bool
    detection_time_ms: float
    guard_triggered: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def is_false_positive(self) -> bool:
        """Return True if benign input was incorrectly blocked."""
        return self.blocked and not self.expected_blocked

    @property
    def is_false_negative(self) -> bool:
        """Return True if malicious input was incorrectly allowed."""
        return not self.blocked and self.expected_blocked

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attack_id": self.attack.id,
            "attack_name": self.attack.name,
            "category": self.attack.category.value,
            "severity": self.attack.severity,
            "blocked": self.blocked,
            "expected_blocked": self.expected_blocked,
            "correct": self.correct,
            "detection_time_ms": self.detection_time_ms,
            "guard_triggered": self.guard_triggered,
            "is_false_positive": self.is_false_positive,
            "is_false_negative": self.is_false_negative,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class RedTeamResults:
    """Aggregated results from a red team test run.

    Attributes:
        results: List of individual attack results.
        start_time: When the test run started.
        end_time: When the test run ended.
        config: Configuration used for the run.
    """
    results: List[AttackResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_attacks(self) -> int:
        """Total number of attacks executed."""
        return len(self.results)

    @property
    def correct_detections(self) -> int:
        """Number of correctly handled attacks."""
        return sum(1 for r in self.results if r.correct)

    @property
    def false_positives(self) -> int:
        """Number of false positives (benign blocked)."""
        return sum(1 for r in self.results if r.is_false_positive)

    @property
    def false_negatives(self) -> int:
        """Number of false negatives (malicious allowed)."""
        return sum(1 for r in self.results if r.is_false_negative)

    @property
    def accuracy(self) -> float:
        """Overall accuracy rate."""
        if self.total_attacks == 0:
            return 0.0
        return self.correct_detections / self.total_attacks

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        blocked = sum(1 for r in self.results if r.blocked)
        if blocked == 0:
            return 1.0
        true_positives = sum(1 for r in self.results if r.blocked and r.expected_blocked)
        return true_positives / blocked

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        should_block = sum(1 for r in self.results if r.expected_blocked)
        if should_block == 0:
            return 1.0
        true_positives = sum(1 for r in self.results if r.blocked and r.expected_blocked)
        return true_positives / should_block

    @property
    def f1_score(self) -> float:
        """F1 Score: harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def avg_detection_time_ms(self) -> float:
        """Average detection time in milliseconds."""
        if not self.results:
            return 0.0
        return sum(r.detection_time_ms for r in self.results) / len(self.results)

    @property
    def duration_seconds(self) -> float:
        """Total test duration in seconds."""
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def get_results_by_category(self) -> Dict[AttackCategory, List[AttackResult]]:
        """Group results by attack category."""
        by_category: Dict[AttackCategory, List[AttackResult]] = {}
        for result in self.results:
            cat = result.attack.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)
        return by_category

    def get_category_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics by category."""
        by_category = self.get_results_by_category()
        stats = {}
        for cat, results in by_category.items():
            total = len(results)
            correct = sum(1 for r in results if r.correct)
            stats[cat.value] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
                "false_positives": sum(1 for r in results if r.is_false_positive),
                "false_negatives": sum(1 for r in results if r.is_false_negative),
            }
        return stats

    def get_failed_attacks(self) -> List[AttackResult]:
        """Get attacks that were not correctly handled."""
        return [r for r in self.results if not r.correct]

    def get_high_severity_failures(self) -> List[AttackResult]:
        """Get high-severity attacks that were not blocked."""
        return [
            r for r in self.results
            if not r.correct and r.attack.severity >= 4
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": {
                "total_attacks": self.total_attacks,
                "correct_detections": self.correct_detections,
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "avg_detection_time_ms": self.avg_detection_time_ms,
                "duration_seconds": self.duration_seconds,
            },
            "category_stats": self.get_category_stats(),
            "results": [r.to_dict() for r in self.results],
            "failed_attacks": [r.to_dict() for r in self.get_failed_attacks()],
            "config": self.config,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class RedTeamRunner:
    """Runner for executing red team attacks against guardrails.

    This class orchestrates the execution of attack payloads against
    the guardrails system, collecting and aggregating results.

    Attributes:
        input_guards: InputGuardrails instance for testing.
        output_guards: OutputGuardrails instance for testing.
        verbose: Whether to print progress during execution.
    """

    def __init__(
        self,
        input_guards: Optional[InputGuardrails] = None,
        output_guards: Optional[OutputGuardrails] = None,
        verbose: bool = False
    ):
        """Initialize the runner.

        Args:
            input_guards: InputGuardrails to test. Creates new if None.
            output_guards: OutputGuardrails to test. Creates new if None.
            verbose: Print progress during execution.
        """
        self.input_guards = input_guards or InputGuardrails()
        self.output_guards = output_guards or OutputGuardrails()
        self.verbose = verbose

    def run_single_attack(self, attack: AttackPayload) -> AttackResult:
        """Execute a single attack and return the result.

        Args:
            attack: The attack payload to test.

        Returns:
            AttackResult with the outcome.
        """
        start_time = time.perf_counter()
        error = None
        blocked = False
        guard_triggered = None
        details = None

        try:
            # Run the attack through input guardrails
            result = self.input_guards.check_prompt_injection(attack.payload)

            blocked = not result.passed
            if blocked:
                guard_triggered = result.guard_name
                details = result.details

        except Exception as e:
            error = str(e)
            blocked = False  # If we error, we didn't block

        end_time = time.perf_counter()
        detection_time_ms = (end_time - start_time) * 1000

        correct = blocked == attack.expected_blocked

        return AttackResult(
            attack=attack,
            blocked=blocked,
            expected_blocked=attack.expected_blocked,
            correct=correct,
            detection_time_ms=detection_time_ms,
            guard_triggered=guard_triggered,
            details=details,
            error=error,
        )

    def run_attacks(
        self,
        attacks: Optional[List[AttackPayload]] = None,
        categories: Optional[List[AttackCategory]] = None,
        min_severity: Optional[int] = None,
        max_workers: int = 1,
    ) -> RedTeamResults:
        """Run a set of attacks against the guardrails.

        Args:
            attacks: Specific attacks to run. Uses all if None.
            categories: Filter to specific categories.
            min_severity: Minimum severity level to include.
            max_workers: Number of parallel workers (1 = sequential).

        Returns:
            RedTeamResults with all attack outcomes.
        """
        # Determine which attacks to run
        if attacks is None:
            attacks = ATTACK_PAYLOADS.copy()

        if categories:
            attacks = [a for a in attacks if a.category in categories]

        if min_severity:
            attacks = [a for a in attacks if a.severity >= min_severity]

        if self.verbose:
            print(f"Running {len(attacks)} attacks...")

        results = RedTeamResults(
            start_time=datetime.now(),
            config={
                "total_attacks": len(attacks),
                "categories": [c.value for c in categories] if categories else "all",
                "min_severity": min_severity,
                "max_workers": max_workers,
            }
        )

        if max_workers == 1:
            # Sequential execution
            for i, attack in enumerate(attacks):
                result = self.run_single_attack(attack)
                results.results.append(result)

                if self.verbose:
                    status = "✓" if result.correct else "✗"
                    print(f"  [{i+1}/{len(attacks)}] {status} {attack.id}: {attack.name}")
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_single_attack, attack): attack
                    for attack in attacks
                }

                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.results.append(result)

                    if self.verbose:
                        status = "✓" if result.correct else "✗"
                        print(f"  [{i+1}/{len(attacks)}] {status} {result.attack.id}")

        results.end_time = datetime.now()

        if self.verbose:
            print(f"\nCompleted in {results.duration_seconds:.2f}s")
            print(f"Accuracy: {results.accuracy:.1%}")
            print(f"False Positives: {results.false_positives}")
            print(f"False Negatives: {results.false_negatives}")

        return results

    def run_category(self, category: AttackCategory) -> RedTeamResults:
        """Run all attacks for a specific category.

        Args:
            category: The attack category to test.

        Returns:
            RedTeamResults for that category.
        """
        attacks = get_attacks_by_category(category)
        return self.run_attacks(attacks=attacks)

    def run_high_severity(self, min_severity: int = 4) -> RedTeamResults:
        """Run only high-severity attacks.

        Args:
            min_severity: Minimum severity level (default 4).

        Returns:
            RedTeamResults for high-severity attacks.
        """
        attacks = get_attacks_by_severity(min_severity)
        return self.run_attacks(attacks=attacks)

    def run_quick_scan(self) -> RedTeamResults:
        """Run a quick scan with one attack per category.

        Useful for rapid testing during development.

        Returns:
            RedTeamResults from the quick scan.
        """
        attacks = []
        for category in AttackCategory:
            category_attacks = get_attacks_by_category(category)
            if category_attacks:
                # Take the highest severity attack from each category
                highest = max(category_attacks, key=lambda a: a.severity)
                attacks.append(highest)

        return self.run_attacks(attacks=attacks)

    def run_with_callback(
        self,
        attacks: Optional[List[AttackPayload]] = None,
        on_result: Optional[Callable[[AttackResult], None]] = None,
    ) -> RedTeamResults:
        """Run attacks with a callback for each result.

        Useful for streaming results to a UI.

        Args:
            attacks: Attacks to run. Uses all if None.
            on_result: Callback called with each result.

        Returns:
            RedTeamResults with all outcomes.
        """
        if attacks is None:
            attacks = ATTACK_PAYLOADS.copy()

        results = RedTeamResults(start_time=datetime.now())

        for attack in attacks:
            result = self.run_single_attack(attack)
            results.results.append(result)
            if on_result:
                on_result(result)

        results.end_time = datetime.now()
        return results
