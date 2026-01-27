"""Example: Basic Portfolio Analysis with the Agent.

This example demonstrates how to use the Portfolio Analysis Agent
to analyze a portfolio programmatically.

Usage:
    python examples/basic_analysis.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import PortfolioAgent
from agent.core import AgentConfig


def main():
    # Create agent with verbose output
    config = AgentConfig(
        provider="claude",
        verbose=True,
        report_output_dir="./reports"
    )
    agent = PortfolioAgent(config)

    # List available tools
    print("Available Tools by Category:")
    print("=" * 50)
    for category, tools in agent.get_available_tools().items():
        print(f"\n{category.upper()}:")
        for tool in tools:
            print(f"  - {tool}")

    # Analyze a portfolio
    print("\n" + "=" * 50)
    print("Running Portfolio Analysis...")
    print("=" * 50)

    response = agent.analyze(
        portfolio_id=102,
        request="""Please analyze this portfolio:
1. Show me the current holdings
2. Calculate VaR at 95% and 99% confidence
3. Analyze factor exposures
4. What market regimes is this portfolio most exposed to?
5. Give me your top 3 recommendations for improving this portfolio""",
        generate_report=True
    )

    print("\n" + "=" * 50)
    print("ANALYSIS RESULT:")
    print("=" * 50)
    print(response.content)

    print("\n" + "=" * 50)
    print("EXECUTION SUMMARY:")
    print("=" * 50)
    print(f"Tools executed: {len(response.tool_executions)}")
    for exec in response.tool_executions:
        status = "OK" if "error" not in exec.result else "ERROR"
        print(f"  - {exec.tool_name}: {status}")

    if response.report_path:
        print(f"\nReport saved to: {response.report_path}")

    if response.charts_generated:
        print(f"Charts generated: {len(response.charts_generated)}")


if __name__ == "__main__":
    main()
