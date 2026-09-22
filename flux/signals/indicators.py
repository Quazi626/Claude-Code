"""Bar-level technical indicators: session VWAP, EMA, rolling volume average.

All functions take/return pandas Series aligned to the input bars' DatetimeIndex.
Bars are expected to have columns: open, high, low, close, volume.
"""
from __future__ import annotations

import pandas as pd


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at the start of each calendar trading session (date)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]

    session_key = df.index.date
    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = df["volume"].groupby(session_key).cumsum()
    return cum_pv / cum_vol.replace(0, pd.NA)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rolling_volume_avg(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Average volume of the PRIOR `lookback` bars (excludes the current bar
    so the "current volume vs avg" comparison isn't comparing a bar to itself).
    """
    return df["volume"].shift(1).rolling(window=lookback, min_periods=lookback).mean()


def add_indicators(
    df: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    volume_lookback: int = 20,
) -> pd.DataFrame:
    out = df.copy()
    out["vwap"] = session_vwap(out)
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["volume_avg"] = rolling_volume_avg(out, volume_lookback)
    return out
