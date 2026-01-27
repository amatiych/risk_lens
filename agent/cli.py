"""Command-line interface for the Portfolio Analysis Agent.

Usage:
    python -m agent.cli --portfolio 102 --request "Analyze risk factors"
    python -m agent.cli --portfolio 102 --full-analysis --report
    python -m agent.cli --list-portfolios
    python -m agent.cli --interactive --portfolio 102
"""

import argparse
import sys
from typing import Optional

from agent.core import PortfolioAgent, AgentConfig, run_analysis
from models.portfolio import Portfolio


def list_portfolios() -> None:
    """List all available portfolios."""
    print("\nAvailable Portfolios:")
    print("-" * 50)

    try:
        portfolios = Portfolio.get_portfolios()
        for portfolio_id, row in portfolios.iterrows():
            print(f"  ID: {portfolio_id:4d} | Name: {row['name']:<20} | NAV: ${row['nav']:,.2f}")
    except Exception as e:
        print(f"Error listing portfolios: {e}")

    print("-" * 50)


def list_tools() -> None:
    """List all available tools."""
    agent = PortfolioAgent()
    tools_by_category = agent.get_available_tools()

    print("\nAvailable Tools:")
    print("-" * 50)

    for category, tools in sorted(tools_by_category.items()):
        print(f"\n{category.upper()}:")
        for tool_name in sorted(tools):
            print(f"  - {tool_name}")

    print("\n" + "-" * 50)


def run_interactive(portfolio_id: Optional[int], provider: str, verbose: bool) -> None:
    """Run the agent in interactive mode."""
    config = AgentConfig(provider=provider, verbose=verbose)
    agent = PortfolioAgent(config)

    print("\nPortfolio Analysis Agent - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  /load <id>  - Load a portfolio")
    print("  /tools      - List available tools")
    print("  /portfolios - List available portfolios")
    print("  /report     - Generate report for current analysis")
    print("  /quit       - Exit")
    print("=" * 50)

    if portfolio_id:
        print(f"\nLoading portfolio {portfolio_id}...")
        response = agent.analyze(portfolio_id, "Load the portfolio and show me a summary of the holdings")
        print(f"\n{response.content}")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/tools":
                list_tools()
                continue

            if user_input.lower() == "/portfolios":
                list_portfolios()
                continue

            if user_input.lower().startswith("/load "):
                try:
                    pid = int(user_input.split()[1])
                    print(f"\nLoading portfolio {pid}...")
                    response = agent.analyze(pid, "Load the portfolio and show me a summary")
                    print(f"\nAgent: {response.content}")
                except (ValueError, IndexError):
                    print("Usage: /load <portfolio_id>")
                continue

            if user_input.lower() == "/report":
                portfolio = agent.registry.get_context("portfolio")
                if portfolio:
                    print("\nGenerating report...")
                    report_path, charts = agent._generate_report(
                        portfolio.portfolio_id,
                        "Analysis report generated from interactive session."
                    )
                    print(f"Report saved to: {report_path}")
                    print(f"Charts generated: {len(charts)}")
                else:
                    print("No portfolio loaded. Use /load <id> first.")
                continue

            # Regular chat message
            print("\nAgent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response.content)

            if response.tool_executions:
                print(f"\n[Executed {len(response.tool_executions)} tool(s)]")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Portfolio Analysis Agent - AI-powered portfolio risk analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --portfolio 102 --request "What are the main risks?"
  %(prog)s --portfolio 102 --full-analysis --report
  %(prog)s --list-portfolios
  %(prog)s --list-tools
  %(prog)s --interactive --portfolio 102
        """
    )

    parser.add_argument(
        "--portfolio", "-p",
        type=int,
        help="Portfolio ID to analyze"
    )

    parser.add_argument(
        "--request", "-r",
        type=str,
        default="Provide a comprehensive risk analysis including VaR, factor exposures, and regime performance.",
        help="Analysis request in natural language"
    )

    parser.add_argument(
        "--full-analysis", "-f",
        action="store_true",
        help="Run full analysis (VaR, factors, regimes, PCA)"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate markdown report with charts"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./reports",
        help="Directory for report output (default: ./reports)"
    )

    parser.add_argument(
        "--provider",
        choices=["claude", "openai"],
        default="claude",
        help="LLM provider to use (default: claude)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output including tool executions"
    )

    parser.add_argument(
        "--list-portfolios",
        action="store_true",
        help="List all available portfolios"
    )

    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available tools"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode"
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_portfolios:
        list_portfolios()
        return

    if args.list_tools:
        list_tools()
        return

    # Handle interactive mode
    if args.interactive:
        run_interactive(args.portfolio, args.provider, args.verbose)
        return

    # Require portfolio ID for analysis
    if not args.portfolio:
        parser.error("--portfolio is required for analysis (or use --list-portfolios, --list-tools, or --interactive)")

    # Build request
    if args.full_analysis:
        request = """Please perform a comprehensive analysis of this portfolio:
1. Load the portfolio and show holdings summary
2. Calculate VaR at 95% and 99% confidence levels
3. Analyze factor exposures
4. Analyze performance across market regimes
5. Run PCA analysis
6. Show correlation matrix
7. Summarize the key risks and provide recommendations"""
    else:
        request = args.request

    # Run analysis
    print(f"\nAnalyzing portfolio {args.portfolio}...")
    print("-" * 50)

    config = AgentConfig(
        provider=args.provider,
        verbose=args.verbose,
        report_output_dir=args.output_dir
    )

    agent = PortfolioAgent(config)
    response = agent.analyze(
        portfolio_id=args.portfolio,
        request=request,
        generate_report=args.report
    )

    print(f"\n{response.content}")

    if response.tool_executions:
        print(f"\n[Executed {len(response.tool_executions)} tool(s)]")

    if response.report_path:
        print(f"\nReport saved to: {response.report_path}")
        print(f"Charts generated: {len(response.charts_generated)}")


if __name__ == "__main__":
    main()
