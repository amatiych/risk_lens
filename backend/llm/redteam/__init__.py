"""Red Team Testing Suite for LLM Guardrails.

This module provides automated adversarial testing to evaluate the robustness
of guardrails against various attack vectors including prompt injection,
jailbreaks, and data extraction attempts.

Usage:
    from backend.llm.redteam import RedTeamRunner, generate_report

    runner = RedTeamRunner(verbose=True)
    results = runner.run_attacks()
    report = generate_report(results)
    print(format_console_report(report))
"""

from backend.llm.redteam.attacks import (
    AttackCategory,
    AttackPayload,
    ATTACK_PAYLOADS,
    get_attacks_by_category,
    get_attacks_by_severity,
    get_attacks_by_tag,
)
from backend.llm.redteam.runner import (
    RedTeamRunner,
    AttackResult,
    RedTeamResults,
)
from backend.llm.redteam.report import (
    SecurityReport,
    generate_report,
    format_markdown_report,
    format_console_report,
    export_json_report,
    export_markdown_report,
)

__all__ = [
    # Attack definitions
    "AttackCategory",
    "AttackPayload",
    "ATTACK_PAYLOADS",
    "get_attacks_by_category",
    "get_attacks_by_severity",
    "get_attacks_by_tag",
    # Runner
    "RedTeamRunner",
    "AttackResult",
    "RedTeamResults",
    # Reports
    "SecurityReport",
    "generate_report",
    "format_markdown_report",
    "format_console_report",
    "export_json_report",
    "export_markdown_report",
]
