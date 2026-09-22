"""Daily floor (classic) pivot points computed from the prior session's OHLC."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloorPivots:
    p: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float

    def as_dict(self) -> dict:
        return {
            "P": self.p,
            "R1": self.r1,
            "R2": self.r2,
            "R3": self.r3,
            "S1": self.s1,
            "S2": self.s2,
            "S3": self.s3,
        }

    def levels_sorted(self) -> list[tuple[str, float]]:
        return sorted(self.as_dict().items(), key=lambda kv: kv[1])


def compute_floor_pivots(high: float, low: float, close: float) -> FloorPivots:
    """Classic floor trader pivots from prior session H/L/C.

    P  = (H + L + C) / 3
    R1 = 2P - L        S1 = 2P - H
    R2 = P + (H - L)   S2 = P - (H - L)
    R3 = H + 2(P - L)  S3 = L - 2(H - P)
    """
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    return FloorPivots(p=p, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3)


def nearest_pivot_level(price: float, pivots: FloorPivots) -> tuple[str, float]:
    """The pivot level closest to `price` (used to detect a reclaim)."""
    return min(pivots.as_dict().items(), key=lambda kv: abs(kv[1] - price))
