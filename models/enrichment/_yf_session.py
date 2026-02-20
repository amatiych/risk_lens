"""Custom curl_cffi session for yfinance to work on cloud servers."""

from curl_cffi.requests import Session

_session = None


def get_yf_session() -> Session:
    global _session
    if _session is None:
        _session = Session(impersonate="chrome")
    return _session
