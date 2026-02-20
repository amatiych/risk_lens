"""Fallback cache for yfinance when Yahoo Finance blocks cloud IPs."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def load_cached_prices(tickers: List[str]) -> Optional[Dict[str, float]]:
    """Try to load prices from cache. Returns None if not all tickers are cached."""
    price_file = CACHE_DIR / "sample_prices.json"
    if not price_file.exists():
        return None
    with open(price_file) as f:
        cached = json.load(f)
    if all(t in cached for t in tickers):
        return {t: cached[t] for t in tickers}
    return None


def load_cached_timeseries(tickers: List[str]) -> Optional[pd.DataFrame]:
    """Try to load time series from cache. Returns None if not all tickers are cached."""
    ts_file = CACHE_DIR / "sample_timeseries.csv"
    if not ts_file.exists():
        return None
    df = pd.read_csv(ts_file, index_col=0, parse_dates=True)
    if all(t in df.columns for t in tickers):
        return df[tickers]
    return None
