# Risk Lens - AI Portfolio Analysis Agent

An AI-powered portfolio risk analysis agent that uses tools to analyze investment portfolios and generate comprehensive reports with charts.

## Features

- **AI-Powered Analysis**: Uses Claude or OpenAI to orchestrate portfolio analysis
- **Flexible Tool System**: Easy-to-extend decorator-based tool registration
- **Comprehensive Risk Analytics**:
  - Value at Risk (VaR) and Expected Shortfall
  - Factor Analysis (Fama-French model)
  - Regime Analysis (performance across market conditions)
  - PCA Analysis (variance decomposition)
  - Correlation Analysis
- **Report Generation**: Markdown reports with embedded charts
- **Interactive Mode**: Chat-based interface for ad-hoc analysis

## Installation

```bash
# Clone and install dependencies
git clone <repo>
cd risk_lens_ag
pdm install
```

## Quick Start

### Command Line

```bash
# List available portfolios
python -m agent --list-portfolios

# Run full analysis on a portfolio
python -m agent --portfolio 102 --full-analysis --report

# Interactive mode
python -m agent --interactive --portfolio 102

# Custom request
python -m agent --portfolio 102 --request "What are the main risks in this portfolio?"
```

### Python API

```python
from agent import PortfolioAgent
from agent.core import AgentConfig

# Create agent
config = AgentConfig(provider="claude", verbose=True)
agent = PortfolioAgent(config)

# Run analysis
response = agent.analyze(
    portfolio_id=102,
    request="Give me a comprehensive risk analysis",
    generate_report=True
)

print(response.content)
print(f"Report saved to: {response.report_path}")
```

## Adding Custom Tools

The agent uses a decorator-based tool registration system that makes it easy to add new tools:

```python
from agent import tool
from typing import Any, Dict, Optional

@tool(
    name="my_custom_analysis",
    description="Description for the AI to understand when to use this tool",
    parameters={
        "param1": {
            "type": "string",
            "description": "Description of the parameter"
        }
    },
    required=["param1"],
    category="analytics"
)
def my_custom_analysis(
    param1: str,
    _context: Optional[Dict[str, Any]] = None,  # Receives portfolio context
    **kwargs
) -> Dict[str, Any]:
    """Your tool implementation."""
    # Access portfolio from context
    portfolio = _context.get("portfolio") if _context else None

    # Do your analysis
    return {
        "result": "your analysis result",
        "interpretation": "human-readable interpretation"
    }
```

## Available Tools

### Portfolio Management
- `list_portfolios` - List all available portfolios
- `load_portfolio` - Load and enrich a portfolio with market data
- `get_portfolio_holdings` - Get detailed holdings list
- `get_sector_breakdown` - Get sector allocation
- `execute_trade` - Execute a trade in the portfolio

### Risk Analytics
- `calculate_var` - Calculate VaR and Expected Shortfall
- `calculate_factor_exposure` - Analyze factor exposures (Fama-French)
- `analyze_regime_performance` - Performance across market regimes
- `run_pca_analysis` - Principal Component Analysis
- `get_correlation_matrix` - Holdings correlation matrix
- `run_full_analysis` - Run all analytics

### Market Data
- `lookup_stock_info` - Get company information
- `get_historical_prices` - Get historical prices around a date
- `get_market_news` - Get recent news for a ticker
- `find_portfolio_diversifiers` - Find low-correlation diversifiers

### Charts
- `generate_holdings_chart` - Holdings allocation chart
- `generate_correlation_heatmap` - Correlation matrix heatmap
- `generate_factor_chart` - Factor exposure chart
- `generate_regime_chart` - Regime performance chart
- `generate_var_chart` - VaR contribution chart
- `generate_pca_chart` - PCA variance explained chart

## Project Structure

```
risk_lens_ag/
├── agent/                      # AI Agent System
│   ├── __init__.py            # Package exports
│   ├── core.py                # PortfolioAgent class
│   ├── cli.py                 # Command-line interface
│   ├── tool_registry.py       # Tool registration system
│   └── tools/                 # Tool implementations
│       ├── analytics.py       # Risk analytics tools
│       ├── market_data.py     # External data tools
│       ├── portfolio_tools.py # Portfolio management
│       └── chart_tools.py     # Chart generation
├── backend/
│   ├── llm/                   # LLM Provider abstraction
│   │   └── providers/         # Claude, OpenAI implementations
│   ├── risk_engine/           # Risk calculations
│   │   ├── var/              # VaR engine
│   │   ├── factor_analysis.py
│   │   ├── regime_analysis.py
│   │   └── portfolio_pca.py
│   └── reporting/            # Report generation
├── models/                    # Data models
│   ├── portfolio.py          # Portfolio class
│   ├── factor_model.py       # Factor model
│   ├── regime_model.py       # Regime model
│   └── enrichment/           # Data enrichers
├── examples/                  # Example scripts
└── tests/                     # Test suite
```

## Configuration

### Environment Variables

```bash
# For Claude (default)
export CLAUDE_API_KEY=your_api_key

# For OpenAI
export OPENAI_API_KEY=your_api_key
```

### Agent Configuration

```python
from agent.core import AgentConfig

config = AgentConfig(
    provider="claude",        # "claude" or "openai"
    max_iterations=20,        # Max tool call iterations
    verbose=False,            # Print debug info
    report_output_dir="./reports"
)
```

## Examples

See the `examples/` directory for more detailed examples:

- `basic_analysis.py` - Basic portfolio analysis
- `custom_tools.py` - Adding custom tools

## License

MIT
