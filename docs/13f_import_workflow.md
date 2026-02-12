# 13F Filing Import Workflow

## Overview

The 13F Import feature allows users to automatically import institutional investor portfolios from SEC 13F filings directly into Risk-Lens for analysis. This is implemented as an **agentic workflow** where an LLM agent orchestrates multiple tools to complete the task.

## What is a 13F Filing?

SEC Form 13F is a quarterly report filed by institutional investment managers with at least $100 million in equity assets under management. It discloses their U.S. equity holdings as of the end of each quarter.

**Key characteristics:**
- Filed within 45 days after quarter end
- Only long equity positions are reported (no shorts, options, or non-US securities)
- Holdings are identified by CUSIP (not ticker symbols)
- Required for hedge funds, mutual funds, pension funds, and other institutional managers

## Architecture

The workflow uses a **single agent with specialized tools** pattern, which aligns with the existing LLM integration in Risk-Lens.

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│                 (Streamlit - Import 13F)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agent13F                               │
│              (LLM-driven orchestrator)                      │
│                                                             │
│  System Prompt: "You are an SEC 13F filing research        │
│  agent. Follow these steps: search → get filing →          │
│  download → convert → analyze"                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ SEC EDGAR   │   │   CUSIP     │   │  Analysis   │
│   Client    │   │   Mapper    │   │  Pipeline   │
└─────────────┘   └─────────────┘   └─────────────┘
```

## Components

### 1. SEC EDGAR Client (`backend/sec/edgar_client.py`)

Handles all communication with the SEC EDGAR API.

**Features:**
- Rate limiting (SEC requires max 10 requests/second)
- Required User-Agent headers for SEC compliance
- Multiple search strategies for finding filers
- XML parsing for 13F Information Tables

**Key Classes:**
- `SECEdgarClient` - Main API client
- `SECFiler` - Filer information (CIK, name)
- `Filing13F` - Filing metadata (accession number, dates)
- `Holding13F` - Individual holding (CUSIP, shares, value)

### 2. CUSIP Mapper (`backend/sec/cusip_mapper.py`)

Converts CUSIP identifiers to ticker symbols.

**Mapping Strategy:**
1. **OpenFIGI API** (primary) - Free, reliable for US equities
2. **Yahoo Finance** (validation) - Confirms ticker is tradeable
3. **Caching** - Avoids repeated lookups for same CUSIPs

**Key Classes:**
- `CUSIPMapper` - Main mapper with batch support
- `CUSIPMapping` - Mapping result with confidence score

### 3. Agent Tools (`backend/llm/tools_13f.py`)

Five tools available to the LLM agent:

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `search_sec_filer` | Find institutional investor by name | Company name | CIK, name, entity type |
| `get_latest_13f_filing` | Get most recent 13F filing | CIK | Accession number, filing date |
| `download_13f_holdings` | Parse 13F Information Table | CIK, accession | List of holdings with CUSIPs |
| `convert_13f_to_portfolio` | Map CUSIPs to tickers | Holdings list | Portfolio CSV |
| `run_13f_analysis` | Execute Risk-Lens analysis | Portfolio CSV | VaR, factors, regimes |

### 4. Agent Orchestrator (`backend/llm/agent_13f.py`)

The `Agent13F` class drives the workflow:

```python
agent = Agent13F(config=LLMConfig(provider="claude"))
result = agent.run_sync("Find Renaissance Technologies' latest 13F", top_n=50)
```

**Agent Behavior:**
- Uses the configured LLM provider (Claude or OpenAI)
- Follows a predefined workflow via system prompt
- Executes tools in sequence based on LLM decisions
- Handles errors and provides status updates
- Returns structured `Agent13FResult`

### 5. Frontend Component (`frontend/components/import_13f.py`)

Streamlit UI for the import workflow:

- Company name input field
- Holdings limit selector (top N by value)
- Progress display during execution
- Filing details preview
- Portfolio preview with unmapped CUSIPs warning
- "Run Full Analysis" button to proceed to analysis

## Workflow Steps

### Step 1: Search for Filer

```
User Input: "Berkshire Hathaway"
     │
     ▼
search_sec_filer("Berkshire Hathaway")
     │
     ▼
SEC EDGAR Full-Text Search API
     │
     ▼
Output: [{cik: "0001067983", name: "BERKSHIRE HATHAWAY INC"}]
```

The search uses SEC's full-text search API to find 13F filers, with a fallback to the company search endpoint.

### Step 2: Get Latest Filing

```
Input: CIK "0001067983"
     │
     ▼
get_latest_13f_filing("0001067983")
     │
     ▼
SEC EDGAR Submissions API: data.sec.gov/submissions/CIK{cik}.json
     │
     ▼
Output: {
  accession_number: "0000950123-24-012345",
  filing_date: "2024-11-14",
  report_date: "2024-09-30"
}
```

### Step 3: Download Holdings

```
Input: CIK, Accession Number
     │
     ▼
download_13f_holdings(cik, accession)
     │
     ▼
1. Get filing index.json to find XML file
2. Download Information Table XML
3. Parse XML to extract holdings
     │
     ▼
Output: [
  {cusip: "037833100", name: "APPLE INC", shares: 400000000, value: 69900000},
  {cusip: "594918104", name: "MICROSOFT CORP", shares: 100000000, value: 40000000},
  ...
]
```

### Step 4: Convert to Portfolio

```
Input: Holdings list (optionally limited to top N)
     │
     ▼
convert_13f_to_portfolio(holdings, top_n=50)
     │
     ▼
For each holding:
  1. Query OpenFIGI API with CUSIP
  2. Get ticker symbol
  3. Validate with yfinance
     │
     ▼
Output: {
  portfolio_csv: "ticker,shares\nAAPL,400000000\nMSFT,100000000\n...",
  statistics: {successful_mappings: 48, failed_mappings: 2},
  unmapped_holdings: [{cusip: "...", name: "...", error: "..."}]
}
```

### Step 5: Run Analysis

```
Input: Portfolio CSV, Portfolio Name
     │
     ▼
run_13f_analysis(csv, "Berkshire Hathaway 13F (2024-09-30)")
     │
     ▼
1. Parse CSV to Portfolio object
2. Enrich with Yahoo Finance prices
3. Run VaR calculation
4. Run factor analysis
5. Run regime analysis
     │
     ▼
Output: {
  summary: {positions: 48, var_95: 0.0234, var_99: 0.0312},
  factor_exposures: {Mkt-RF: 1.05, SMB: -0.12, HML: 0.34},
  ...
}
```

## Data Flow Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  SEC EDGAR   │────▶│   13F XML    │────▶│   Holdings   │
│     API      │     │   (CUSIPs)   │     │    List      │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   OpenFIGI   │────▶│   Tickers    │────▶│  Portfolio   │
│     API      │     │              │     │     CSV      │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│Yahoo Finance │────▶│   Prices &   │────▶│  Analysis    │
│     API      │     │ Time Series  │     │   Results    │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Filer not found | Return suggestions, explain 13F requirements |
| No 13F filings | Explain entity may not file 13F |
| XML parse error | Try alternative parsing strategies |
| CUSIP unmapped | Skip position, report in results, continue |
| Rate limited | Exponential backoff, retry |
| Network timeout | Retry with backoff |
| Stale filing (>6 months) | Warn user but proceed |

## Usage Example

### Via UI

1. Navigate to "Import 13F" in the sidebar
2. Enter institutional investor name (e.g., "Renaissance Technologies")
3. Set maximum holdings (default: 50)
4. Click "Search & Import"
5. Review the filing details and portfolio preview
6. Click "Run Full Analysis" to proceed

### Via Code

```python
from backend.llm.agent_13f import Agent13F
from backend.llm.providers import LLMConfig

# Initialize agent
config = LLMConfig(provider="claude")
agent = Agent13F(config=config)

# Run workflow
result = agent.run_sync(
    "Find and analyze the latest 13F filing for Bridgewater Associates",
    top_n=100
)

if result.success:
    print(f"Portfolio: {result.portfolio_name}")
    print(f"Holdings: {result.mapped_tickers}")
    print(f"Unmapped: {len(result.unmapped_cusips)}")

    # Access the portfolio CSV
    portfolio_csv = result.portfolio_csv

    # Access analysis results
    analysis = result.analysis_results
```

## Configuration

### SEC EDGAR User-Agent

The SEC requires a User-Agent header with contact information. This is configured in `backend/sec/edgar_client.py`:

```python
SEC_USER_AGENT = "Risk-Lens aleksey@purpleswansoftware.com"
```

### Rate Limiting

SEC EDGAR enforces a limit of 10 requests/second. The client automatically rate-limits to 100ms between requests.

### LLM Provider

The agent uses the same LLM provider configured in the UI sidebar (Claude or OpenAI).

## Limitations

1. **Long positions only** - 13F filings don't include short positions
2. **US equities only** - Non-US securities are not in 13F
3. **45-day delay** - Filings are due 45 days after quarter end
4. **CUSIP mapping** - Some securities (derivatives, private placements) may not map to tickers
5. **Large portfolios** - Very large portfolios may take time to process due to CUSIP mapping

## API Reference

### SECEdgarClient

```python
from backend.sec.edgar_client import SECEdgarClient

client = SECEdgarClient()

# Search for filer
filers = client.search_filer("Berkshire Hathaway")

# Get filings
filings = client.get_13f_filings(cik="0001067983", limit=5)

# Download holdings
holdings = client.get_13f_holdings(cik, accession_number)
```

### CUSIPMapper

```python
from backend.sec.cusip_mapper import CUSIPMapper

mapper = CUSIPMapper()

# Single mapping
result = mapper.map_cusip("037833100", "APPLE INC")
print(result.ticker)  # "AAPL"

# Batch mapping (more efficient)
results = mapper.map_batch([
    {"cusip": "037833100", "name": "APPLE INC"},
    {"cusip": "594918104", "name": "MICROSOFT CORP"},
])
```

### Agent13F

```python
from backend.llm.agent_13f import Agent13F, Agent13FResult

agent = Agent13F()

# With status updates (generator)
for status in agent.run("Find Renaissance Technologies' 13F", top_n=50):
    print(status)
# Generator returns Agent13FResult when complete

# Synchronous (simpler)
result: Agent13FResult = agent.run_sync("Find Renaissance Technologies' 13F")
```
