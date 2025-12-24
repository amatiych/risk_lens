"""Tests for the red team testing suite."""

import pytest
from datetime import datetime

from backend.llm.redteam import (
    AttackCategory,
    AttackPayload,
    ATTACK_PAYLOADS,
    get_attacks_by_category,
    get_attacks_by_severity,
    get_attacks_by_tag,
    RedTeamRunner,
    AttackResult,
    RedTeamResults,
    SecurityReport,
    generate_report,
    format_markdown_report,
    format_console_report,
)


class TestAttackPayloads:
    """Tests for attack payload definitions."""

    def test_attack_payloads_not_empty(self):
        """Should have attack payloads defined."""
        assert len(ATTACK_PAYLOADS) > 0

    def test_all_categories_have_attacks(self):
        """Each category should have at least one attack."""
        for category in AttackCategory:
            attacks = get_attacks_by_category(category)
            assert len(attacks) > 0, f"Category {category} has no attacks"

    def test_attack_payload_has_required_fields(self):
        """Each attack should have all required fields."""
        for attack in ATTACK_PAYLOADS:
            assert attack.id, "Attack missing id"
            assert attack.category, "Attack missing category"
            assert attack.name, "Attack missing name"
            assert attack.description, "Attack missing description"
            assert attack.payload, "Attack missing payload"
            assert 1 <= attack.severity <= 5, f"Invalid severity: {attack.severity}"

    def test_attack_ids_are_unique(self):
        """All attack IDs should be unique."""
        ids = [a.id for a in ATTACK_PAYLOADS]
        assert len(ids) == len(set(ids)), "Duplicate attack IDs found"

    def test_get_attacks_by_category(self):
        """Should filter attacks by category."""
        injection_attacks = get_attacks_by_category(AttackCategory.PROMPT_INJECTION)
        assert all(a.category == AttackCategory.PROMPT_INJECTION for a in injection_attacks)

    def test_get_attacks_by_severity(self):
        """Should filter attacks by minimum severity."""
        high_severity = get_attacks_by_severity(4)
        assert all(a.severity >= 4 for a in high_severity)

    def test_get_attacks_by_tag(self):
        """Should filter attacks by tag."""
        basic_attacks = get_attacks_by_tag("basic")
        assert all("basic" in (a.tags or []) for a in basic_attacks)


class TestAttackResult:
    """Tests for AttackResult dataclass."""

    @pytest.fixture
    def sample_attack(self):
        """Create a sample attack for testing."""
        return AttackPayload(
            id="TEST001",
            category=AttackCategory.PROMPT_INJECTION,
            name="Test Attack",
            description="A test attack",
            payload="Test payload",
            expected_blocked=True,
            severity=3,
        )

    def test_attack_result_creation(self, sample_attack):
        """Should create AttackResult with all fields."""
        result = AttackResult(
            attack=sample_attack,
            blocked=True,
            expected_blocked=True,
            correct=True,
            detection_time_ms=5.0,
        )
        assert result.blocked is True
        assert result.correct is True

    def test_false_positive_detection(self, sample_attack):
        """Should correctly identify false positives."""
        sample_attack.expected_blocked = False
        result = AttackResult(
            attack=sample_attack,
            blocked=True,
            expected_blocked=False,
            correct=False,
            detection_time_ms=5.0,
        )
        assert result.is_false_positive is True
        assert result.is_false_negative is False

    def test_false_negative_detection(self, sample_attack):
        """Should correctly identify false negatives."""
        result = AttackResult(
            attack=sample_attack,
            blocked=False,
            expected_blocked=True,
            correct=False,
            detection_time_ms=5.0,
        )
        assert result.is_false_negative is True
        assert result.is_false_positive is False

    def test_to_dict(self, sample_attack):
        """Should convert to dictionary."""
        result = AttackResult(
            attack=sample_attack,
            blocked=True,
            expected_blocked=True,
            correct=True,
            detection_time_ms=5.0,
        )
        d = result.to_dict()
        assert d["attack_id"] == "TEST001"
        assert d["blocked"] is True
        assert d["correct"] is True


class TestRedTeamResults:
    """Tests for RedTeamResults aggregation."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        attacks = [
            AttackPayload(
                id=f"TEST{i:03d}",
                category=AttackCategory.PROMPT_INJECTION,
                name=f"Test Attack {i}",
                description="Test",
                payload="Test",
                expected_blocked=True,
                severity=3,
            )
            for i in range(5)
        ]

        results = RedTeamResults(start_time=datetime.now())

        # 3 correct, 2 false negatives
        for i, attack in enumerate(attacks):
            blocked = i < 3
            results.results.append(AttackResult(
                attack=attack,
                blocked=blocked,
                expected_blocked=True,
                correct=blocked,
                detection_time_ms=float(i),
            ))

        results.end_time = datetime.now()
        return results

    def test_total_attacks(self, sample_results):
        """Should count total attacks."""
        assert sample_results.total_attacks == 5

    def test_correct_detections(self, sample_results):
        """Should count correct detections."""
        assert sample_results.correct_detections == 3

    def test_false_negatives(self, sample_results):
        """Should count false negatives."""
        assert sample_results.false_negatives == 2

    def test_accuracy(self, sample_results):
        """Should calculate accuracy."""
        assert sample_results.accuracy == 0.6  # 3/5

    def test_precision(self, sample_results):
        """Should calculate precision."""
        # 3 blocked, all 3 should be blocked = 100%
        assert sample_results.precision == 1.0

    def test_recall(self, sample_results):
        """Should calculate recall."""
        # 5 should be blocked, 3 were = 60%
        assert sample_results.recall == 0.6

    def test_avg_detection_time(self, sample_results):
        """Should calculate average detection time."""
        # (0+1+2+3+4)/5 = 2.0
        assert sample_results.avg_detection_time_ms == 2.0

    def test_to_dict(self, sample_results):
        """Should convert to dictionary."""
        d = sample_results.to_dict()
        assert "summary" in d
        assert "category_stats" in d
        assert "results" in d

    def test_to_json(self, sample_results):
        """Should convert to JSON."""
        json_str = sample_results.to_json()
        assert "summary" in json_str
        assert "accuracy" in json_str


class TestRedTeamRunner:
    """Tests for RedTeamRunner execution."""

    @pytest.fixture
    def runner(self):
        """Create a runner for testing."""
        return RedTeamRunner(verbose=False)

    def test_run_single_attack(self, runner):
        """Should run a single attack and return result."""
        attack = ATTACK_PAYLOADS[0]
        result = runner.run_single_attack(attack)

        assert isinstance(result, AttackResult)
        assert result.attack == attack
        assert result.detection_time_ms >= 0

    def test_run_attacks_all(self, runner):
        """Should run all attacks."""
        results = runner.run_attacks()

        assert isinstance(results, RedTeamResults)
        assert results.total_attacks == len(ATTACK_PAYLOADS)

    def test_run_attacks_filtered_by_category(self, runner):
        """Should filter by category."""
        categories = [AttackCategory.PROMPT_INJECTION]
        results = runner.run_attacks(categories=categories)

        expected_count = len(get_attacks_by_category(AttackCategory.PROMPT_INJECTION))
        assert results.total_attacks == expected_count

    def test_run_attacks_filtered_by_severity(self, runner):
        """Should filter by severity."""
        results = runner.run_attacks(min_severity=5)

        expected_count = len(get_attacks_by_severity(5))
        assert results.total_attacks == expected_count

    def test_run_quick_scan(self, runner):
        """Should run one attack per category."""
        results = runner.run_quick_scan()

        # Should have at most one per category
        assert results.total_attacks <= len(AttackCategory)

    def test_run_category(self, runner):
        """Should run attacks for specific category."""
        results = runner.run_category(AttackCategory.JAILBREAK)

        for result in results.results:
            assert result.attack.category == AttackCategory.JAILBREAK

    def test_run_high_severity(self, runner):
        """Should run only high severity attacks."""
        results = runner.run_high_severity(min_severity=4)

        for result in results.results:
            assert result.attack.severity >= 4

    def test_run_with_callback(self, runner):
        """Should call callback for each result."""
        called = []

        def on_result(result):
            called.append(result)

        attacks = ATTACK_PAYLOADS[:3]
        results = runner.run_with_callback(attacks=attacks, on_result=on_result)

        assert len(called) == 3
        assert results.total_attacks == 3


class TestSecurityReport:
    """Tests for security report generation."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        runner = RedTeamRunner(verbose=False)
        return runner.run_quick_scan()

    def test_generate_report(self, sample_results):
        """Should generate a security report."""
        report = generate_report(sample_results)

        assert isinstance(report, SecurityReport)
        assert report.title == "Guardrails Security Assessment"
        assert report.generated_at is not None
        assert len(report.recommendations) > 0

    def test_report_to_dict(self, sample_results):
        """Should convert report to dictionary."""
        report = generate_report(sample_results)
        d = report.to_dict()

        assert "title" in d
        assert "summary" in d
        assert "category_breakdown" in d
        assert "vulnerabilities" in d
        assert "recommendations" in d

    def test_format_markdown_report(self, sample_results):
        """Should format as markdown."""
        report = generate_report(sample_results)
        md = format_markdown_report(report)

        assert "# Guardrails Security Assessment" in md
        assert "## Executive Summary" in md
        assert "## Category Breakdown" in md
        assert "## Recommendations" in md

    def test_format_console_report(self, sample_results):
        """Should format for console output."""
        report = generate_report(sample_results)
        console = format_console_report(report)

        assert "RISK LEVEL" in console
        assert "SUMMARY" in console
        assert "RESULTS BY CATEGORY" in console


class TestIntegration:
    """Integration tests for the full red team workflow."""

    def test_full_workflow(self):
        """Should execute full red team workflow."""
        # Create runner
        runner = RedTeamRunner(verbose=False)

        # Run attacks
        results = runner.run_attacks(min_severity=4)

        # Generate report
        report = generate_report(results)

        # Verify results
        assert results.total_attacks > 0
        assert report.recommendations is not None

        # Verify report formats
        md = format_markdown_report(report)
        console = format_console_report(report)

        assert len(md) > 0
        assert len(console) > 0

    def test_guardrails_block_known_attacks(self):
        """Known basic attacks should be blocked."""
        runner = RedTeamRunner(verbose=False)

        # Test only prompt injection category which has well-known patterns
        results = runner.run_category(AttackCategory.PROMPT_INJECTION)

        # For prompt injection, we expect higher accuracy since these
        # are well-known patterns
        blocked_count = sum(1 for r in results.results if r.blocked)
        total = results.total_attacks

        if total > 0:
            block_rate = blocked_count / total
            # Prompt injection should have decent block rate
            assert block_rate >= 0.4, (
                f"Prompt injection block rate too low: {block_rate:.1%}"
            )

    def test_red_team_provides_useful_metrics(self):
        """Red team suite should provide useful security metrics."""
        runner = RedTeamRunner(verbose=False)
        results = runner.run_attacks()

        # Verify we get meaningful metrics
        assert results.accuracy > 0, "Should have some detections"
        assert results.precision >= 0, "Precision should be valid"
        assert results.recall >= 0, "Recall should be valid"

        # Verify category breakdown works
        stats = results.get_category_stats()
        assert len(stats) > 0, "Should have category stats"

        # Verify we identify failures for improvement
        failed = results.get_failed_attacks()
        assert isinstance(failed, list), "Should return list of failures"
