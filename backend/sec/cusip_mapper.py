"""CUSIP to ticker symbol mapping.

Handles conversion between CUSIP identifiers (used in 13F filings)
and ticker symbols (used by yfinance and the portfolio model).

Strategies:
1. OpenFIGI API (free, reliable for US equities)
2. Yahoo Finance ticker validation
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import yfinance as yf


@dataclass
class CUSIPMapping:
    """CUSIP to ticker mapping result."""

    cusip: str
    ticker: Optional[str]
    name: str
    confidence: float  # 0.0 to 1.0
    source: str  # "openfigi", "yfinance", "cache"
    error: Optional[str] = None


@dataclass
class CUSIPMapper:
    """Convert CUSIPs to ticker symbols.

    Uses OpenFIGI API as primary source with yfinance validation.
    Maintains a cache to avoid repeated lookups.
    """

    _cache: Dict[str, CUSIPMapping] = field(default_factory=dict)
    _openfigi_url: str = "https://api.openfigi.com/v3/mapping"
    _rate_limit_delay: float = 0.1  # 100ms between API calls

    def map_cusip(self, cusip: str, name: str = "") -> CUSIPMapping:
        """Map a single CUSIP to ticker.

        Args:
            cusip: 9-character CUSIP identifier.
            name: Optional issuer name for fallback matching.

        Returns:
            CUSIPMapping with ticker if found, or error if not.
        """
        # Normalize CUSIP
        cusip = cusip.strip().upper()

        # Check cache first
        if cusip in self._cache:
            cached = self._cache[cusip]
            return CUSIPMapping(
                cusip=cusip,
                ticker=cached.ticker,
                name=cached.name or name,
                confidence=cached.confidence,
                source="cache",
            )

        # Try OpenFIGI
        ticker = self._try_openfigi(cusip)
        if ticker:
            # Validate with yfinance
            if self._validate_with_yfinance(ticker):
                mapping = CUSIPMapping(
                    cusip=cusip,
                    ticker=ticker,
                    name=name,
                    confidence=0.95,
                    source="openfigi",
                )
                self._cache[cusip] = mapping
                return mapping

        # Try name-based search with yfinance as fallback
        if name:
            ticker = self._try_name_search(name)
            if ticker:
                mapping = CUSIPMapping(
                    cusip=cusip,
                    ticker=ticker,
                    name=name,
                    confidence=0.7,
                    source="yfinance",
                )
                self._cache[cusip] = mapping
                return mapping

        # Failed to map
        mapping = CUSIPMapping(
            cusip=cusip,
            ticker=None,
            name=name,
            confidence=0.0,
            source="none",
            error="Could not map CUSIP to ticker",
        )
        self._cache[cusip] = mapping
        return mapping

    def map_batch(
        self, holdings: List[Dict]
    ) -> List[Tuple[Dict, CUSIPMapping]]:
        """Map multiple CUSIPs in batch (more efficient).

        Uses OpenFIGI batch API to reduce API calls.

        Args:
            holdings: List of holding dicts with 'cusip' and optional 'name' keys.

        Returns:
            List of tuples (original_holding, CUSIPMapping).
        """
        results = []

        # First, check cache and collect uncached CUSIPs
        uncached = []
        for holding in holdings:
            cusip = holding.get("cusip", "").strip().upper()
            name = holding.get("name", "")

            if cusip in self._cache:
                cached = self._cache[cusip]
                results.append(
                    (
                        holding,
                        CUSIPMapping(
                            cusip=cusip,
                            ticker=cached.ticker,
                            name=cached.name or name,
                            confidence=cached.confidence,
                            source="cache",
                        ),
                    )
                )
            else:
                uncached.append((holding, cusip, name))

        if not uncached:
            return results

        # Batch lookup via OpenFIGI (max 100 per request)
        batch_size = 100
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            cusip_list = [item[1] for item in batch]

            figi_results = self._batch_openfigi(cusip_list)

            for (holding, cusip, name), figi_result in zip(batch, figi_results):
                if figi_result and self._validate_with_yfinance(figi_result):
                    mapping = CUSIPMapping(
                        cusip=cusip,
                        ticker=figi_result,
                        name=name,
                        confidence=0.95,
                        source="openfigi",
                    )
                else:
                    # Try name-based fallback
                    ticker = self._try_name_search(name) if name else None
                    if ticker:
                        mapping = CUSIPMapping(
                            cusip=cusip,
                            ticker=ticker,
                            name=name,
                            confidence=0.7,
                            source="yfinance",
                        )
                    else:
                        mapping = CUSIPMapping(
                            cusip=cusip,
                            ticker=None,
                            name=name,
                            confidence=0.0,
                            source="none",
                            error="Could not map CUSIP to ticker",
                        )

                self._cache[cusip] = mapping
                results.append((holding, mapping))

            # Rate limit between batches
            if i + batch_size < len(uncached):
                time.sleep(self._rate_limit_delay)

        return results

    def _try_openfigi(self, cusip: str) -> Optional[str]:
        """Use OpenFIGI API for single CUSIP lookup.

        Args:
            cusip: CUSIP to look up.

        Returns:
            Ticker symbol if found, None otherwise.
        """
        try:
            time.sleep(self._rate_limit_delay)

            response = requests.post(
                self._openfigi_url,
                headers={"Content-Type": "application/json"},
                json=[{"idType": "ID_CUSIP", "idValue": cusip}],
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                if "data" in result and len(result["data"]) > 0:
                    # Prefer US equity ticker
                    for item in result["data"]:
                        if item.get("exchCode") in (
                            "US",
                            "UN",
                            "UW",
                            "UA",
                            "UR",
                        ):
                            return item.get("ticker")
                    # Fall back to first ticker
                    return result["data"][0].get("ticker")

        except (requests.RequestException, KeyError, IndexError):
            pass

        return None

    def _batch_openfigi(self, cusips: List[str]) -> List[Optional[str]]:
        """Batch CUSIP lookup via OpenFIGI.

        Args:
            cusips: List of CUSIPs to look up.

        Returns:
            List of tickers (or None) in same order as input.
        """
        try:
            time.sleep(self._rate_limit_delay)

            jobs = [{"idType": "ID_CUSIP", "idValue": c} for c in cusips]

            response = requests.post(
                self._openfigi_url,
                headers={"Content-Type": "application/json"},
                json=jobs,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data:
                if "data" in item and len(item["data"]) > 0:
                    # Prefer US equity ticker
                    ticker = None
                    for entry in item["data"]:
                        if entry.get("exchCode") in (
                            "US",
                            "UN",
                            "UW",
                            "UA",
                            "UR",
                        ):
                            ticker = entry.get("ticker")
                            break
                    if not ticker:
                        ticker = item["data"][0].get("ticker")
                    results.append(ticker)
                else:
                    results.append(None)

            return results

        except (requests.RequestException, KeyError, IndexError):
            return [None] * len(cusips)

    def _validate_with_yfinance(self, ticker: str) -> bool:
        """Validate ticker exists in yfinance.

        Args:
            ticker: Ticker symbol to validate.

        Returns:
            True if ticker is valid and has data.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Check if we got meaningful data
            return info.get("regularMarketPrice") is not None or info.get(
                "previousClose"
            ) is not None
        except Exception:
            return False

    def _try_name_search(self, name: str) -> Optional[str]:
        """Try to find ticker by company name using yfinance search.

        Args:
            name: Company name to search.

        Returns:
            Ticker if found, None otherwise.
        """
        # Clean up the name for search
        name_clean = name.strip()
        if not name_clean:
            return None

        # Common suffixes to try removing
        suffixes = [
            " INC",
            " CORP",
            " CO",
            " LTD",
            " LLC",
            " PLC",
            " CLASS A",
            " CLASS B",
            " CLASS C",
            " CL A",
            " CL B",
            " COM",
            " COMMON",
            " NEW",
        ]

        search_names = [name_clean]
        for suffix in suffixes:
            if name_clean.upper().endswith(suffix):
                search_names.append(name_clean[: -len(suffix)].strip())

        # Try each variation
        for search_name in search_names[:3]:  # Limit attempts
            try:
                # Use yfinance search (via info lookup with partial match)
                # This is a heuristic - try common ticker patterns
                potential_ticker = (
                    search_name.split()[0][:5].upper().replace(".", "")
                )
                if self._validate_with_yfinance(potential_ticker):
                    return potential_ticker
            except Exception:
                continue

        return None

    def get_stats(self) -> Dict:
        """Get mapping statistics.

        Returns:
            Dict with counts of successful/failed mappings.
        """
        total = len(self._cache)
        successful = sum(1 for m in self._cache.values() if m.ticker)
        failed = total - successful

        return {
            "total_cached": total,
            "successful_mappings": successful,
            "failed_mappings": failed,
            "success_rate": successful / total if total > 0 else 0.0,
        }
