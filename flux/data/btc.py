"""BTC/USD spot price fetch via Alpaca's crypto market data, with caching.

Used solely for the FLUX BTC gate (block MSTR/COIN longs when BTC is below a
configured threshold).
"""
from __future__ import annotations

import time

from flux.data.alpaca_client import FluxAlpacaClient

_cache: dict[str, tuple[float, float]] = {}


def get_btc_price(client: FluxAlpacaClient, symbol: str = "BTC/USD", refresh_seconds: int = 60) -> float:
    now = time.time()
    cached = _cache.get(symbol)
    if cached and (now - cached[1]) < refresh_seconds:
        return cached[0]

    price = client.get_latest_crypto_price(symbol)
    _cache[symbol] = (price, now)
    return price


def reset_cache() -> None:
    _cache.clear()
