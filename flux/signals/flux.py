"""The FLUX signal: price vs VWAP + 9/21 EMA alignment + volume confirmation
+ a pivot-level reclaim, evaluated on intraday bars.

Long call:  price > VWAP AND EMA9 > EMA21 AND volume > avg*mult AND price
            just reclaimed a pivot level from below.
Long put:   the mirror image (price < VWAP, EMA9 < EMA21, breakdown through
            a pivot level from above).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from flux.data.pivots import FloorPivots
from flux.signals.indicators import add_indicators


@dataclass
class FluxSignal:
    symbol: str
    timestamp: datetime
    direction: Optional[str]  # "call", "put", or None
    price: float
    vwap: float
    ema_fast: float
    ema_slow: float
    volume: float
    volume_avg: float
    pivot_level_name: Optional[str]
    pivot_level_value: Optional[float]
    reasons: dict = field(default_factory=dict)

    def as_log_row(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction or "",
            "price": self.price,
            "vwap": self.vwap,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "volume": self.volume,
            "volume_avg": self.volume_avg,
            "pivot_level_name": self.pivot_level_name or "",
            "pivot_level_value": self.pivot_level_value or "",
            "above_vwap": self.reasons.get("above_vwap"),
            "below_vwap": self.reasons.get("below_vwap"),
            "ema_bull": self.reasons.get("ema_bull"),
            "ema_bear": self.reasons.get("ema_bear"),
            "volume_confirm": self.reasons.get("volume_confirm"),
            "pivot_reclaim_up": self.reasons.get("pivot_reclaim_up"),
            "pivot_reclaim_down": self.reasons.get("pivot_reclaim_down"),
        }


def _find_pivot_crossing(
    prev_close: float,
    curr_close: float,
    pivots: FloorPivots,
    buffer_pct: float,
    direction: str,
) -> Optional[tuple[str, float]]:
    """Find a pivot level the price just crossed on this bar.

    direction="up":   prev_close was below the level, curr_close is at/above it
    direction="down":  prev_close was above the level, curr_close is at/below it
    `buffer_pct` (e.g. 0.05) allows the crossing to be "close enough" rather
    than requiring an exact tick through the level.
    """
    best: Optional[tuple[str, float]] = None
    best_dist = None
    for name, level in pivots.as_dict().items():
        buffer = level * (buffer_pct / 100.0)
        if direction == "up":
            crossed = prev_close < (level - buffer) and curr_close >= (level - buffer)
        else:
            crossed = prev_close > (level + buffer) and curr_close <= (level + buffer)
        if crossed:
            dist = abs(curr_close - level)
            if best is None or dist < best_dist:
                best = (name, level)
                best_dist = dist
    return best


def evaluate_signal(
    symbol: str,
    bars: pd.DataFrame,
    pivots: FloorPivots,
    ema_fast_span: int = 9,
    ema_slow_span: int = 21,
    volume_lookback: int = 20,
    volume_multiplier: float = 1.0,
    pivot_reclaim_buffer_pct: float = 0.05,
    lookback_bars_for_cross: int = 3,
) -> Optional[FluxSignal]:
    """Evaluate the most recent completed bar in `bars` for a FLUX entry signal.

    Returns None if there isn't enough history yet to evaluate indicators.
    """
    if len(bars) < max(ema_slow_span, volume_lookback) + 1:
        return None

    df = add_indicators(
        bars,
        ema_fast=ema_fast_span,
        ema_slow=ema_slow_span,
        volume_lookback=volume_lookback,
    )
    curr = df.iloc[-1]
    if pd.isna(curr["volume_avg"]) or pd.isna(curr["vwap"]):
        return None

    above_vwap = bool(curr["close"] > curr["vwap"])
    below_vwap = bool(curr["close"] < curr["vwap"])
    ema_bull = bool(curr["ema_fast"] > curr["ema_slow"])
    ema_bear = bool(curr["ema_fast"] < curr["ema_slow"])
    volume_confirm = bool(curr["volume"] > curr["volume_avg"] * volume_multiplier)

    window = df.iloc[-(lookback_bars_for_cross + 1) :]
    reclaim_up = None
    reclaim_down = None
    for i in range(1, len(window)):
        prev_c = window["close"].iloc[i - 1]
        curr_c = window["close"].iloc[i]
        up = _find_pivot_crossing(prev_c, curr_c, pivots, pivot_reclaim_buffer_pct, "up")
        down = _find_pivot_crossing(prev_c, curr_c, pivots, pivot_reclaim_buffer_pct, "down")
        if up:
            reclaim_up = up
        if down:
            reclaim_down = down

    direction = None
    pivot_name, pivot_value = None, None
    if above_vwap and ema_bull and volume_confirm and reclaim_up:
        direction = "call"
        pivot_name, pivot_value = reclaim_up
    elif below_vwap and ema_bear and volume_confirm and reclaim_down:
        direction = "put"
        pivot_name, pivot_value = reclaim_down

    return FluxSignal(
        symbol=symbol,
        timestamp=curr.name.to_pydatetime() if hasattr(curr.name, "to_pydatetime") else curr.name,
        direction=direction,
        price=float(curr["close"]),
        vwap=float(curr["vwap"]),
        ema_fast=float(curr["ema_fast"]),
        ema_slow=float(curr["ema_slow"]),
        volume=float(curr["volume"]),
        volume_avg=float(curr["volume_avg"]),
        pivot_level_name=pivot_name,
        pivot_level_value=pivot_value,
        reasons={
            "above_vwap": above_vwap,
            "below_vwap": below_vwap,
            "ema_bull": ema_bull,
            "ema_bear": ema_bear,
            "volume_confirm": volume_confirm,
            "pivot_reclaim_up": bool(reclaim_up),
            "pivot_reclaim_down": bool(reclaim_down),
        },
    )
