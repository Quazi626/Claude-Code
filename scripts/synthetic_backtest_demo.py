#!/usr/bin/env python3
"""Offline mechanical demo of the backtest engine, using FABRICATED price
data instead of real Alpaca/yfinance history (this sandbox has no network
access to either). This proves the code path — signal generation, VIX
sizing, BTC gate, contract selection math, Black-Scholes pricing, exits,
fees/slippage, and metrics reporting — executes correctly end to end.

THE NUMBERS BELOW ARE NOT A REAL PERFORMANCE RESULT. The price data is
synthetic and deliberately engineered to produce some clean trend days so
the FLUX signal actually fires; it says nothing about how the strategy
would perform on real SPY price action. For a real backtest, run
`python -m flux.backtest.run --start ... --end ...` with your own Alpaca
keys on a machine with network access.
"""
from __future__ import annotations

import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flux.backtest.engine import run_backtest
from flux.backtest.metrics import compute_metrics
from flux.config import load_config
from flux.data.pivots import compute_floor_pivots

ET = pytz.timezone("America/New_York")
rng = np.random.default_rng(7)

SYMBOL = "SPY"
N_TRADING_DAYS = 65  # roughly the last three months
BASE_PRICE = 450.0
BASE_VOLUME = 8000.0  # per 5-min bar


def session_bar_times(day) -> list[datetime]:
    start = ET.localize(datetime.combine(day, dtime(9, 30)))
    return [start + timedelta(minutes=5 * i) for i in range(79)]  # 9:30 -> 16:00


def _bars_from_closes(times, closes, volumes) -> pd.DataFrame:
    closes = np.asarray(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    noise = rng.uniform(0.01, 0.05, len(closes))
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.maximum(volumes, 1).astype(int)},
        index=pd.DatetimeIndex(times),
    )


def generate_trend_day(day, target_level: float, direction: str, start_price: float) -> pd.DataFrame:
    """A day engineered to cleanly reclaim `target_level` (a pivot) with a
    volume spike on the crossing bars, then continue trending so VWAP/EMA
    stay aligned with the breakout — i.e. a textbook FLUX entry.
    """
    times = session_bar_times(day)
    n = len(times)
    pre_n, cross_n = 25, 5
    post_n = n - pre_n - cross_n
    side = -1 if direction == "call" else 1  # call: approach level from below

    closes = np.empty(n)
    closes[0] = target_level + side * max(abs(start_price - target_level), 1.0) * 0.5
    for i in range(1, pre_n):
        closes[i] = closes[i - 1] + rng.normal(0, 0.04)

    ramp_target = target_level - side * max(abs(start_price - target_level) * 0.6, 1.5)
    closes[pre_n : pre_n + cross_n] = np.linspace(closes[pre_n - 1], ramp_target, cross_n)

    drift = -side * rng.uniform(0.02, 0.07, post_n)
    closes[pre_n + cross_n :] = closes[pre_n + cross_n - 1] + np.cumsum(drift)

    volumes = np.concatenate([
        rng.normal(BASE_VOLUME, BASE_VOLUME * 0.1, pre_n),
        rng.uniform(BASE_VOLUME * 2.2, BASE_VOLUME * 3.0, cross_n),
        rng.normal(BASE_VOLUME * 1.4, BASE_VOLUME * 0.2, post_n),
    ])
    return _bars_from_closes(times, closes, volumes)


def generate_flat_day(day, start_price: float) -> pd.DataFrame:
    times = session_bar_times(day)
    n = len(times)
    closes = start_price + np.cumsum(rng.normal(0, 0.035, n))
    volumes = rng.normal(BASE_VOLUME, BASE_VOLUME * 0.15, n)
    return _bars_from_closes(times, closes, volumes)


def build_synthetic_dataset(start: datetime, symbol_base_price: float):
    trading_days = [d.date() for d in pd.bdate_range(start, periods=N_TRADING_DAYS)]

    pattern_cycle = ["flat", "flat", "flat", "call", "flat", "flat", "put", "flat"]

    prior_ohlc = {
        "open": symbol_base_price, "high": symbol_base_price * 1.004,
        "low": symbol_base_price * 0.996, "close": symbol_base_price,
    }
    daily_rows: dict = {}
    all_bars = []
    vix_values: dict = {}

    for idx, day in enumerate(trading_days):
        daily_rows[day] = dict(prior_ohlc)
        pivots = compute_floor_pivots(prior_ohlc["high"], prior_ohlc["low"], prior_ohlc["close"])

        pattern = pattern_cycle[idx % len(pattern_cycle)]
        if pattern == "call":
            day_df = generate_trend_day(day, pivots.r1, "call", prior_ohlc["close"])
        elif pattern == "put":
            day_df = generate_trend_day(day, pivots.s1, "put", prior_ohlc["close"])
        else:
            day_df = generate_flat_day(day, prior_ohlc["close"])

        all_bars.append(day_df)
        prior_ohlc = {
            "open": float(day_df["open"].iloc[0]), "high": float(day_df["high"].max()),
            "low": float(day_df["low"].min()), "close": float(day_df["close"].iloc[-1]),
        }
        # Sine wave sweeping ~10-38 so all three VIX sizing tiers get exercised.
        vix_values[day] = float(np.clip(22 + 14 * np.sin(2 * np.pi * idx / 14), 10, 38))

    bars = pd.concat(all_bars).sort_index()
    daily_ohlc = pd.DataFrame.from_dict(daily_rows, orient="index")
    vix_daily = pd.Series(vix_values)
    return bars, daily_ohlc, vix_daily


def main():
    print("=" * 70)
    print("SYNTHETIC backtest demo — FABRICATED data, mechanical smoke test only")
    print("=" * 70)

    cfg = load_config()
    start = datetime.now() - timedelta(days=N_TRADING_DAYS * 1.45)  # ~3 calendar months back
    bars, daily_ohlc, vix_daily = build_synthetic_dataset(start, BASE_PRICE)

    print(f"\nGenerated {len(bars)} bars across {daily_ohlc.shape[0]} trading days for {SYMBOL}")
    print(f"VIX range in synthetic data: {vix_daily.min():.1f} - {vix_daily.max():.1f}")

    trades = run_backtest(SYMBOL, bars, daily_ohlc, vix_daily, cfg, starting_equity=25000.0)
    metrics = compute_metrics(trades, starting_equity=25000.0)

    print(f"\n{len(trades)} simulated trades:")
    for t in trades[:15]:
        print(
            f"  {t.entry_time.date()} {t.direction:4s} strike={t.strike:6.1f} "
            f"dte={t.dte_at_entry} contracts={t.contracts} "
            f"entry=${t.entry_price:.2f} exit=${t.exit_price:.2f} "
            f"net_pnl=${t.net_pnl:8.2f} reason={t.exit_reason}"
        )
    if len(trades) > 15:
        print(f"  ... and {len(trades) - 15} more")

    print("\n" + "=" * 70)
    print("METRICS (synthetic data — not a real performance estimate)")
    print("=" * 70)
    print(metrics.summary())

    if not trades:
        print(
            "\nNo trades were generated — the engine ran without errors, but the "
            "synthetic data didn't trip a FLUX signal. This can still happen "
            "depending on random noise in the 'flat' days; try re-running."
        )


if __name__ == "__main__":
    main()
