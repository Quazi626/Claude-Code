"""Historical data assembly for the backtester: intraday underlying bars,
prior-day OHLC for pivots, and daily VIX/BTC series, all localized to
America/New_York (the risk rules' time gates compare wall-clock ET times).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from flux.data.alpaca_client import FluxAlpacaClient

ET = "America/New_York"


def _localize_et(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def load_intraday_bars(
    client: FluxAlpacaClient, symbol: str, start: date, end: date, timeframe: str = "5Min"
) -> pd.DataFrame:
    df = client.get_stock_bars(
        symbol, timeframe, start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    return _localize_et(df)


def load_daily_ohlc(client: FluxAlpacaClient, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily bars indexed by plain `date`, used as the PRIOR day's OHLC for
    each trading day's floor pivots (the caller looks up `day` and reads that
    row as "prior session" input, so this frame is naturally shifted by
    reading yesterday's row when computing today's pivots).
    """
    df = client.get_stock_bars(
        symbol, "1Day",
        start=datetime.combine(start - timedelta(days=10), datetime.min.time()),
        end=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    if df.empty:
        return df
    df = df.copy()
    df.index = [ts.date() if hasattr(ts, "date") else ts for ts in df.index]
    # Shift so row for `day` holds the PRIOR day's OHLC.
    shifted = df.shift(1)
    shifted.index = df.index
    return shifted.dropna()


def load_vix_daily(start: date, end: date) -> pd.Series:
    hist = yf.Ticker("^VIX").history(start=start - timedelta(days=10), end=end + timedelta(days=1), interval="1d")
    series = hist["Close"]
    series.index = [ts.date() for ts in series.index]
    return series


def load_btc_daily(client: FluxAlpacaClient, start: date, end: date) -> pd.Series:
    req = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        timeframe=TimeFrame.Day,
        start=datetime.combine(start - timedelta(days=10), datetime.min.time()),
        end=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    df = client.crypto_data.get_crypto_bars(req).df
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("BTC/USD", level=0)
    series = df["close"]
    series.index = [ts.date() if hasattr(ts, "date") else ts for ts in series.index]
    return series
