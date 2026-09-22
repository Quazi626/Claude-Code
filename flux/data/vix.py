"""VIX level fetch via yfinance, with a small in-process cache.

Alpaca does not carry the CBOE VIX index, so this uses yfinance's free ^VIX
feed as a separate, independent data source per the project's design
decision (see README).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import yfinance as yf

log = logging.getLogger("flux.data.vix")

_cache: dict[str, tuple[float, float]] = {}  # symbol -> (value, fetched_at_epoch)


def get_vix_level(symbol: str = "^VIX", refresh_seconds: int = 60) -> float:
    now = time.time()
    cached = _cache.get(symbol)
    if cached and (now - cached[1]) < refresh_seconds:
        return cached[0]

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d", interval="1m")
    if hist.empty:
        hist = ticker.history(period="5d", interval="1d")
    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    value = float(hist["Close"].iloc[-1])
    _cache[symbol] = (value, now)
    return value


def reset_cache() -> None:
    _cache.clear()
