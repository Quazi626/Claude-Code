"""Black-Scholes European option pricing, used to simulate option premiums
from historical underlying price bars when no historical options chain is
available (see README for why this approach was chosen over a paid
historical options data feed).
"""
from __future__ import annotations

import math

from scipy.stats import norm


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    iv: float,
    option_type: str,
) -> float:
    """European option price. At/after expiry (t_years <= 0) this collapses
    to intrinsic value, which is what a 0DTE contract's price converges to
    into the close anyway.
    """
    if t_years <= 0 or iv <= 0:
        if option_type == "call":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * math.exp(-rate * t_years) * norm.cdf(d2)
    else:
        price = strike * math.exp(-rate * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return max(price, 0.0)
