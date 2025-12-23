import pandas as pd
from typing import List
from backend.risk_engine.var.var_engine import VaR


def format_currency(value: float) -> str:
    """Format a value as currency."""
    if abs(value) >= 1e6:
        return f"${value/1e6:,.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:,.2f}K"
    return f"${value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_var_table(var_results: List[VaR]) -> pd.DataFrame:
    """Format VaR results as a DataFrame."""
    rows = []
    for var in var_results:
        rows.append({
            'Confidence Level': format_percentage(var.ci),
            'VaR': format_percentage(var.var),
            'Expected Shortfall': format_percentage(abs(var.es)),
            'VaR Date': var.var_date.strftime('%Y-%m-%d')
        })
    return pd.DataFrame(rows)


def format_marginal_var_table(var_results: List[VaR], tickers: List[str]) -> pd.DataFrame:
    """Format marginal/incremental VaR as DataFrame."""
    rows = []
    for var in var_results:
        for ticker, mvar, ivar in zip(tickers, var.marginal_var, var.incremental_var):
            rows.append({
                'CI': format_percentage(var.ci),
                'Ticker': ticker,
                'Marginal VaR': format_percentage(mvar),
                'Incremental VaR': format_percentage(ivar)
            })
    return pd.DataFrame(rows)


def format_holdings_table(holdings: pd.DataFrame) -> pd.DataFrame:
    """Format holdings for display."""
    df = holdings.copy()
    display_df = pd.DataFrame({
        'Ticker': df.index,
        'Shares': df['shares'].values,
        'Price': [format_currency(p) for p in df['price'].values],
        'Market Value': [format_currency(mv) for mv in df['market_value'].values],
        'Weight': [format_percentage(w) for w in df['weight'].values]
    })
    return display_df


def format_factor_table(factor_result) -> pd.DataFrame:
    """Format factor analysis results."""
    rows = []
    for factor, beta, risk_pct in zip(
        factor_result.factors,
        factor_result.betas,
        factor_result.marginal_risk
    ):
        rows.append({
            'Factor': factor,
            'Beta': f"{beta:.4f}",
            'Risk Contribution': format_percentage(risk_pct)
        })
    return pd.DataFrame(rows)
