"""FLUX risk rules: time gates, VIX-based sizing, the BTC gate, session trade
limits, daily profit/loss stop, per-trade position sizing, and per-position
exit rules.

Everything here is a pure function or a small stateful dataclass operating on
plain Python values (no Alpaca SDK objects), so it can be unit tested in
isolation and reused identically by both the live bot and the backtester.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from enum import Enum
from typing import Optional

import pytz

ET = pytz.timezone("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def parse_hhmm(value: str) -> dtime:
    hh, mm = value.split(":")
    return dtime(hour=int(hh), minute=int(mm))


# ---------------------------------------------------------------------------
# Time gates
# ---------------------------------------------------------------------------

def is_entry_allowed(now: datetime, no_entries_before: dtime) -> bool:
    """No new entries before the configured cutoff (e.g. 9:45 AM ET).

    `now` is expected to already be localized to the trading timezone (e.g.
    via `now_et()`); `.time()` then gives that timezone's wall-clock time.
    """
    return now.time() >= no_entries_before


def is_force_close_time(now: datetime, force_close_by: dtime) -> bool:
    """True once it's time to flatten all 0DTE positions (e.g. 3:15 PM ET)."""
    return now.time() >= force_close_by


# ---------------------------------------------------------------------------
# VIX-based sizing
# ---------------------------------------------------------------------------

class VixTier(str, Enum):
    FULL = "full"
    HALF = "half"
    EXTREME = "extreme"


@dataclass(frozen=True)
class VixSizing:
    tier: VixTier
    size_multiplier: float
    allow_0dte: bool
    min_dte: int
    max_dte: Optional[int]  # None = no upper bound imposed by the VIX rule


def classify_vix(
    vix_level: float,
    full_size_max: float = 25.0,
    half_size_max: float = 30.0,
    half_size_min_dte: int = 1,
    half_size_max_dte: int = 3,
    extreme_min_dte: int = 1,
    half_size_multiplier: float = 0.5,
) -> VixSizing:
    """VIX < 25: full size, 0DTE allowed.
    25 <= VIX < 30: half size, DTE must be within [half_size_min_dte, half_size_max_dte].
    VIX >= 30: half size, 0DTE blocked, DTE >= extreme_min_dte.
    """
    if vix_level < full_size_max:
        return VixSizing(VixTier.FULL, 1.0, True, 0, None)
    if vix_level < half_size_max:
        return VixSizing(
            VixTier.HALF, half_size_multiplier, False, half_size_min_dte, half_size_max_dte
        )
    return VixSizing(VixTier.EXTREME, half_size_multiplier, False, extreme_min_dte, None)


# ---------------------------------------------------------------------------
# BTC gate
# ---------------------------------------------------------------------------

def btc_gate_allows(
    symbol: str,
    btc_price: float,
    min_price_usd: float,
    gated_symbols: tuple[str, ...] = ("MSTR", "COIN"),
) -> bool:
    """Block MSTR/COIN long entries when BTC is below the configured floor."""
    if symbol.upper() not in {s.upper() for s in gated_symbols}:
        return True
    return btc_price >= min_price_usd


# ---------------------------------------------------------------------------
# Session-level trade limits & daily profit/loss stop
# ---------------------------------------------------------------------------

@dataclass
class SessionRiskState:
    max_trades_per_session: int = 5
    daily_profit_target: float = 500.0
    daily_max_loss: float = 500.0
    trades_taken: int = 0
    realized_pnl: float = 0.0
    _stopped_reason: Optional[str] = field(default=None, repr=False)

    def _check_daily_stop(self) -> Optional[str]:
        if self.realized_pnl >= self.daily_profit_target:
            return "daily_profit_target_hit"
        if self.realized_pnl <= -abs(self.daily_max_loss):
            return "daily_max_loss_hit"
        return None

    def record_realized_pnl(self, pnl_delta: float) -> None:
        """Apply a realized P&L delta (e.g. from a closed trade). Once the
        daily target or max loss is breached, trading stays stopped for the
        rest of the session even if a later delta moves P&L back inside the
        band (the stop is a one-way latch for the day).
        """
        self.realized_pnl += pnl_delta
        reason = self._check_daily_stop()
        if reason and not self._stopped_reason:
            self._stopped_reason = reason

    def record_trade_opened(self) -> None:
        self.trades_taken += 1

    def can_open_new_trade(self) -> tuple[bool, Optional[str]]:
        if self._stopped_reason:
            return False, self._stopped_reason
        if self.trades_taken >= self.max_trades_per_session:
            return False, "max_trades_per_session_reached"
        return True, None

    @property
    def stopped_reason(self) -> Optional[str]:
        return self._stopped_reason

    def reset_for_new_session(self) -> None:
        self.trades_taken = 0
        self.realized_pnl = 0.0
        self._stopped_reason = None


# ---------------------------------------------------------------------------
# Per-trade position sizing
# ---------------------------------------------------------------------------

def calc_contracts(
    account_equity: float,
    risk_pct_per_trade: float,
    vix_size_multiplier: float,
    option_price_per_share: float,
    max_contracts_per_trade: int,
    contract_multiplier: int = 100,
) -> int:
    """Number of contracts such that total premium at risk stays within the
    per-trade risk cap (% of equity), scaled by the VIX size multiplier.
    """
    if option_price_per_share <= 0 or account_equity <= 0:
        return 0
    risk_dollars = account_equity * (risk_pct_per_trade / 100.0) * vix_size_multiplier
    contract_cost = option_price_per_share * contract_multiplier
    if contract_cost <= 0:
        return 0
    contracts = int(risk_dollars // contract_cost)
    return max(0, min(contracts, max_contracts_per_trade))


# ---------------------------------------------------------------------------
# Per-position exit rules
# ---------------------------------------------------------------------------

def should_exit_position(
    entry_price: float,
    current_price: float,
    entry_time: datetime,
    now: datetime,
    stop_loss_pct: float,
    profit_target_pct: float,
    time_stop_minutes: int,
) -> Optional[str]:
    """Returns the exit reason ("stop_loss" / "profit_target" / "time_stop")
    or None if the position should stay open. Evaluated in that priority
    order so a bar that satisfies more than one condition reports the
    risk-control reason (stop loss) ahead of the time stop.
    """
    if entry_price <= 0:
        return None
    pnl_pct = (current_price - entry_price) / entry_price * 100.0
    if pnl_pct <= -abs(stop_loss_pct):
        return "stop_loss"
    if pnl_pct >= profit_target_pct:
        return "profit_target"
    elapsed_minutes = (now - entry_time).total_seconds() / 60.0
    if elapsed_minutes >= time_stop_minutes:
        return "time_stop"
    return None
