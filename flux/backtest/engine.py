"""Single-symbol backtest engine: replays historical intraday bars through
the same FLUX signal/risk logic the live bot uses, simulating option
premiums with Black-Scholes and applying fees + slippage on every fill.

Scope note: this simulates one underlying's option P&L per run (equity
curve = starting capital + cumulative realized option P&L for that symbol).
Run it once per watchlist symbol and aggregate `Trade` lists for a
portfolio-level report (see `flux/backtest/metrics.py` and
`flux/backtest/run.py`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from flux.backtest.black_scholes import bs_price
from flux.data.pivots import compute_floor_pivots
from flux.risk.rules import (
    SessionRiskState,
    btc_gate_allows,
    calc_contracts,
    classify_vix,
    is_entry_allowed,
    is_force_close_time,
    parse_hhmm,
    should_exit_position,
)
from flux.signals.flux import evaluate_signal

log = logging.getLogger("flux.backtest.engine")


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    strike: float
    dte_at_entry: int
    contracts: int
    entry_price: float  # per-share premium, after slippage, before fees
    exit_price: float
    fees_paid: float
    exit_reason: str

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.contracts * 100

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees_paid


@dataclass
class _OpenPosition:
    symbol: str
    direction: str
    entry_time: datetime
    strike: float
    expiration: date
    contracts: int
    entry_price: float
    entry_fees: float


def _strike_from_spot(spot: float, increment: float = 1.0) -> float:
    return round(round(spot / increment) * increment, 2)


def run_backtest(
    symbol: str,
    bars: pd.DataFrame,
    daily_ohlc: pd.DataFrame,
    vix_daily: pd.Series,
    cfg,
    btc_daily: Optional[pd.Series] = None,
    starting_equity: float = 25000.0,
) -> list[Trade]:
    """Replay `bars` (intraday OHLCV, DatetimeIndex, tz-aware in ET) for one
    symbol day by day, generating FLUX signals, sizing/gating per the risk
    rules, and simulating fills/exits with Black-Scholes + fees/slippage.

    `daily_ohlc` must have one row per prior trading day with open/high/low/close,
    indexed by date, used to compute each day's floor pivots.
    `vix_daily` / `btc_daily`: Series indexed by date with the prior close.
    """
    schedule = cfg.get("schedule", default={})
    no_entries_before = parse_hhmm(schedule.get("no_entries_before", "09:45"))
    force_close_by = parse_hhmm(schedule.get("force_close_0dte_by", "15:15"))

    sig_cfg = cfg.get("signals", default={})
    vix_cfg = cfg.get("vix", default={})
    btc_cfg = cfg.get("btc_gate", default={})
    sizing_cfg = cfg.get("sizing", default={})
    session_cfg = cfg.get("session_limits", default={})
    exits_cfg = cfg.get("exits", default={})
    bt_cfg = cfg.get("backtest", default={})
    dte_cfg = cfg.get("dte", default={})
    contracts_cfg = cfg.get("contracts", default={})

    fees_per_contract = float(bt_cfg.get("fees_per_contract_usd", 0.65))
    slippage_pct = float(bt_cfg.get("slippage_pct", 2.0))
    iv_assumption = float(bt_cfg.get("iv_assumption", 0.35))
    rate = float(bt_cfg.get("risk_free_rate", 0.05))

    trades: list[Trade] = []
    equity = starting_equity
    session_state = SessionRiskState(
        max_trades_per_session=int(session_cfg.get("max_trades_per_session", 5)),
        daily_profit_target=float(session_cfg.get("daily_profit_target_usd", 500.0)),
        daily_max_loss=float(session_cfg.get("daily_max_loss_usd", 500.0)),
    )

    open_pos: Optional[_OpenPosition] = None
    bars = bars.sort_index()
    dates = sorted(set(bars.index.date))

    for day in dates:
        day_bars = bars[bars.index.date == day]
        if day not in daily_ohlc.index:
            continue
        prior = daily_ohlc.loc[day]
        pivots = compute_floor_pivots(float(prior["high"]), float(prior["low"]), float(prior["close"]))

        vix_level = float(vix_daily.get(day, vix_daily.iloc[vix_daily.index.get_indexer([day], method="pad")[0]]))
        vix_sizing = classify_vix(
            vix_level,
            full_size_max=float(vix_cfg.get("full_size_max", 25.0)),
            half_size_max=float(vix_cfg.get("half_size_max", 30.0)),
            half_size_min_dte=int(vix_cfg.get("half_size_min_dte", 1)),
            half_size_max_dte=int(vix_cfg.get("half_size_max_dte", 3)),
            extreme_min_dte=int(vix_cfg.get("extreme_min_dte", 1)),
            half_size_multiplier=float(sizing_cfg.get("half_size_multiplier", 0.5)),
        )

        btc_ok = True
        if symbol.upper() in {s.upper() for s in btc_cfg.get("gated_symbols", [])} and btc_daily is not None and len(btc_daily) > 0:
            btc_price = float(btc_daily.get(day, btc_daily.iloc[btc_daily.index.get_indexer([day], method="pad")[0]]))
            btc_ok = btc_gate_allows(
                symbol, btc_price, float(btc_cfg.get("min_price_usd", 71000)),
                tuple(btc_cfg.get("gated_symbols", ["MSTR", "COIN"])),
            )

        session_state.reset_for_new_session()

        for i in range(len(day_bars)):
            now = day_bars.index[i].to_pydatetime()
            window = day_bars.iloc[: i + 1]
            spot = float(window["close"].iloc[-1])

            if open_pos is not None:
                t_years = max((datetime.combine(open_pos.expiration, datetime.min.time()) - now).days, 0) / 365.0
                current_price = bs_price(spot, open_pos.strike, t_years, rate, iv_assumption, open_pos.direction)

                force_close = open_pos.expiration == day and is_force_close_time(now, force_close_by)
                exit_reason = should_exit_position(
                    entry_price=open_pos.entry_price,
                    current_price=current_price,
                    entry_time=open_pos.entry_time,
                    now=now,
                    stop_loss_pct=float(exits_cfg.get("stop_loss_pct", 40.0)),
                    profit_target_pct=float(exits_cfg.get("profit_target_pct", 60.0)),
                    time_stop_minutes=int(exits_cfg.get("time_stop_minutes", 120)),
                )
                if force_close and not exit_reason:
                    exit_reason = "force_close_0dte"

                if exit_reason:
                    fill_price = current_price * (1 - slippage_pct / 100.0)
                    fill_price = max(fill_price, 0.0)
                    exit_fees = fees_per_contract * open_pos.contracts
                    trade = Trade(
                        symbol=symbol,
                        direction=open_pos.direction,
                        entry_time=open_pos.entry_time,
                        exit_time=now,
                        strike=open_pos.strike,
                        dte_at_entry=(open_pos.expiration - open_pos.entry_time.date()).days,
                        contracts=open_pos.contracts,
                        entry_price=open_pos.entry_price,
                        exit_price=fill_price,
                        fees_paid=open_pos.entry_fees + exit_fees,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)
                    equity += trade.net_pnl
                    session_state.record_realized_pnl(trade.net_pnl)
                    open_pos = None

            can_trade, _ = session_state.can_open_new_trade()
            if (
                open_pos is None
                and btc_ok
                and can_trade
                and is_entry_allowed(now, no_entries_before)
                and not is_force_close_time(now, force_close_by)
            ):
                signal = evaluate_signal(
                    symbol,
                    window,
                    pivots,
                    ema_fast_span=int(sig_cfg.get("ema_fast", 9)),
                    ema_slow_span=int(sig_cfg.get("ema_slow", 21)),
                    volume_lookback=int(sig_cfg.get("volume_avg_lookback", 20)),
                    volume_multiplier=float(sig_cfg.get("volume_multiplier", 1.0)),
                    pivot_reclaim_buffer_pct=float(sig_cfg.get("pivot_reclaim_buffer_pct", 0.05)),
                )
                if signal and signal.direction:
                    dte = vix_sizing.min_dte if vix_sizing.min_dte else int(dte_cfg.get("default", 0))
                    if not vix_sizing.allow_0dte:
                        dte = max(dte, vix_sizing.min_dte)
                    expiration = day + timedelta(days=dte)
                    strike = _strike_from_spot(spot)
                    t_years = max((expiration - day).days, 0) / 365.0
                    theo_price = bs_price(spot, strike, t_years, rate, iv_assumption, signal.direction)
                    entry_price = theo_price * (1 + slippage_pct / 100.0)

                    contracts = calc_contracts(
                        equity,
                        float(sizing_cfg.get("risk_pct_per_trade", 1.0)),
                        vix_sizing.size_multiplier,
                        entry_price,
                        int(sizing_cfg.get("max_contracts_per_trade", 10)),
                    )
                    if contracts > 0 and entry_price > 0:
                        entry_fees = fees_per_contract * contracts
                        open_pos = _OpenPosition(
                            symbol=symbol,
                            direction=signal.direction,
                            entry_time=now,
                            strike=strike,
                            expiration=expiration,
                            contracts=contracts,
                            entry_price=entry_price,
                            entry_fees=entry_fees,
                        )
                        session_state.record_trade_opened()

        if open_pos is not None:
            spot = float(day_bars["close"].iloc[-1])
            now = day_bars.index[-1].to_pydatetime()
            t_years = max((open_pos.expiration - day).days, 0) / 365.0
            current_price = bs_price(spot, open_pos.strike, t_years, rate, iv_assumption, open_pos.direction)
            fill_price = max(current_price * (1 - slippage_pct / 100.0), 0.0)
            exit_fees = fees_per_contract * open_pos.contracts
            trade = Trade(
                symbol=symbol,
                direction=open_pos.direction,
                entry_time=open_pos.entry_time,
                exit_time=now,
                strike=open_pos.strike,
                dte_at_entry=(open_pos.expiration - open_pos.entry_time.date()).days,
                contracts=open_pos.contracts,
                entry_price=open_pos.entry_price,
                exit_price=fill_price,
                fees_paid=open_pos.entry_fees + exit_fees,
                exit_reason="end_of_day_close",
            )
            trades.append(trade)
            equity += trade.net_pnl
            session_state.record_realized_pnl(trade.net_pnl)
            open_pos = None

    return trades
