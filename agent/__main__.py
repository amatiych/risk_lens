"""Allow running the agent as a module.

Usage:
    python -m agent --portfolio 102 --full-analysis
    python -m agent --interactive
"""

from agent.cli import main

if __name__ == "__main__":
    main()
