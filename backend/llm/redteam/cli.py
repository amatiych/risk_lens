#!/usr/bin/env python3
"""CLI for running red team tests against guardrails.

Usage:
    python -m backend.llm.redteam.cli [OPTIONS]

Examples:
    # Run all attacks
    python -m backend.llm.redteam.cli

    # Run quick scan (one attack per category)
    python -m backend.llm.redteam.cli --quick

    # Run specific categories
    python -m backend.llm.redteam.cli --category prompt_injection --category jailbreak

    # Run high-severity only
    python -m backend.llm.redteam.cli --min-severity 4

    # Export report to file
    python -m backend.llm.redteam.cli --output report.md --format markdown
"""

import argparse
import sys
from typing import List, Optional

from backend.llm.redteam import (
    RedTeamRunner,
    AttackCategory,
    generate_report,
    format_console_report,
    format_markdown_report,
    export_json_report,
    export_markdown_report,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Red Team Testing Suite for LLM Guardrails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all attacks:
    python -m backend.llm.redteam.cli

  Quick scan (one attack per category):
    python -m backend.llm.redteam.cli --quick

  Test specific categories:
    python -m backend.llm.redteam.cli -c prompt_injection -c jailbreak

  High-severity attacks only:
    python -m backend.llm.redteam.cli --min-severity 4

  Export to file:
    python -m backend.llm.redteam.cli -o report.md -f markdown
        """,
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick scan (one attack per category)",
    )

    parser.add_argument(
        "--category", "-c",
        action="append",
        dest="categories",
        choices=[c.value for c in AttackCategory],
        help="Limit to specific attack categories (can be repeated)",
    )

    parser.add_argument(
        "--min-severity",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Minimum severity level to include",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for report",
    )

    parser.add_argument(
        "--format", "-f",
        choices=["console", "markdown", "json"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show progress during execution",
    )

    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available attack categories and exit",
    )

    parser.add_argument(
        "--list-attacks",
        action="store_true",
        help="List all available attacks and exit",
    )

    return parser.parse_args()


def list_categories() -> None:
    """Print available attack categories."""
    print("\nAvailable Attack Categories:")
    print("-" * 40)

    from backend.llm.redteam import ATTACK_PAYLOADS

    # Count attacks per category
    counts = {}
    for attack in ATTACK_PAYLOADS:
        cat = attack.category.value
        counts[cat] = counts.get(cat, 0) + 1

    for category in AttackCategory:
        count = counts.get(category.value, 0)
        print(f"  {category.value:<25} ({count} attacks)")

    print()


def list_attacks() -> None:
    """Print all available attacks."""
    print("\nAvailable Attacks:")
    print("=" * 60)

    from backend.llm.redteam import ATTACK_PAYLOADS

    current_category = None
    for attack in sorted(ATTACK_PAYLOADS, key=lambda a: (a.category.value, a.id)):
        if attack.category != current_category:
            current_category = attack.category
            print(f"\n{current_category.value.upper()}")
            print("-" * 40)

        severity_bar = "●" * attack.severity + "○" * (5 - attack.severity)
        blocked = "should block" if attack.expected_blocked else "should allow"
        print(f"  {attack.id}: {attack.name}")
        print(f"       Severity: {severity_bar} ({attack.severity}/5)")
        print(f"       Expected: {blocked}")
        print()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle listing commands
    if args.list_categories:
        list_categories()
        return 0

    if args.list_attacks:
        list_attacks()
        return 0

    # Run tests
    runner = RedTeamRunner(verbose=args.verbose)

    print("\n" + "=" * 60)
    print("  RISK LENS RED TEAM TESTING SUITE")
    print("=" * 60 + "\n")

    if args.quick:
        print("Running quick scan...")
        results = runner.run_quick_scan()
    elif args.categories:
        categories = [AttackCategory(c) for c in args.categories]
        print(f"Running attacks for categories: {', '.join(args.categories)}")
        results = runner.run_attacks(
            categories=categories,
            min_severity=args.min_severity,
            max_workers=args.parallel,
        )
    elif args.min_severity:
        print(f"Running attacks with severity >= {args.min_severity}")
        results = runner.run_attacks(
            min_severity=args.min_severity,
            max_workers=args.parallel,
        )
    else:
        print("Running all attacks...")
        results = runner.run_attacks(max_workers=args.parallel)

    # Generate report
    report = generate_report(results)

    # Output results
    if args.output:
        if args.format == "json":
            export_json_report(report, args.output)
            print(f"\nReport exported to: {args.output}")
        elif args.format == "markdown":
            export_markdown_report(report, args.output)
            print(f"\nReport exported to: {args.output}")
        else:
            with open(args.output, "w") as f:
                f.write(format_console_report(report))
            print(f"\nReport exported to: {args.output}")
    else:
        if args.format == "markdown":
            print(format_markdown_report(report))
        else:
            print(format_console_report(report))

    # Return exit code based on results
    if results.get_high_severity_failures():
        return 2  # Critical failures
    elif results.false_negatives > 0:
        return 1  # Some failures
    else:
        return 0  # All passed


if __name__ == "__main__":
    sys.exit(main())
