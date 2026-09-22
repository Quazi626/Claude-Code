"""FLUX live/paper trading loop.

Run with:  python -m flux.main [--config config.yaml]

Paper mode is the default and requires only ALPACA_API_KEY/ALPACA_SECRET_KEY
pointed at the paper endpoint. Live trading additionally requires
mode.environment: live in config.yaml, LIVE_TRADING_CONFIRM=YES in the
environment, AND a typed confirmation at startup (see flux.config).
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime

from flux.config import load_config, require_live_confirmation
from flux.data.alpaca_client import FluxAlpacaClient
from flux.data.btc import get_btc_price
from flux.data.pivots import FloorPivots, compute_floor_pivots
from flux.data.vix import get_vix_level
from flux.execution import kill_switch
from flux.execution.options_selector import select_contract
from flux.execution.order_manager import submit_entry_limit_order, submit_exit_limit_order
from flux.logging_utils import FluxLoggers, setup_console_logging
from flux.risk.rules import (
    SessionRiskState,
    btc_gate_allows,
    calc_contracts,
    classify_vix,
    is_entry_allowed,
    is_force_close_time,
    now_et,
    parse_hhmm,
    should_exit_position,
)
from flux.signals.flux import evaluate_signal

log = logging.getLogger("flux.main")


@dataclass
class OpenPosition:
    symbol: str
    option_symbol: str
    direction: str
    contracts: int
    entry_price: float
    entry_time: datetime
    expiration: date
    strike: float


def _get_option_mid(client: FluxAlpacaClient, underlying: str, expiration: date, option_symbol: str):
    snapshot = client.get_option_chain_snapshot(underlying, expiration_date=expiration)
    snap = snapshot.get(option_symbol) if snapshot else None
    if not snap or not snap.latest_quote:
        return None
    bid = float(snap.latest_quote.bid_price or 0)
    ask = float(snap.latest_quote.ask_price or 0)
    if bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def run(cfg) -> None:
    require_live_confirmation(cfg)
    client = FluxAlpacaClient.from_creds(cfg.creds)
    loggers = FluxLoggers(cfg)

    watchlist = cfg.get("watchlist", default=[])
    schedule = cfg.get("schedule", default={})
    no_entries_before = parse_hhmm(schedule.get("no_entries_before", "09:45"))
    force_close_by = parse_hhmm(schedule.get("force_close_0dte_by", "15:15"))

    sig_cfg = cfg.get("signals", default={})
    vix_cfg = cfg.get("vix", default={})
    btc_cfg = cfg.get("btc_gate", default={})
    dte_cfg = cfg.get("dte", default={})
    contracts_cfg = cfg.get("contracts", default={})
    sizing_cfg = cfg.get("sizing", default={})
    session_cfg = cfg.get("session_limits", default={})
    exits_cfg = cfg.get("exits", default={})
    orders_cfg = cfg.get("orders", default={})
    kill_cfg = cfg.get("kill_switch", default={})

    bar_timeframe = sig_cfg.get("bar_timeframes", ["5Min"])[0]
    poll_seconds = int(kill_cfg.get("poll_seconds", 2))
    loop_seconds = max(poll_seconds, int(orders_cfg.get("loop_seconds", 15)))

    session_state = SessionRiskState(
        max_trades_per_session=int(session_cfg.get("max_trades_per_session", 5)),
        daily_profit_target=float(session_cfg.get("daily_profit_target_usd", 500.0)),
        daily_max_loss=float(session_cfg.get("daily_max_loss_usd", 500.0)),
    )
    open_positions: dict[str, OpenPosition] = {}
    pivots_by_symbol: dict[str, FloorPivots] = {}
    current_day: date | None = None

    log.info("FLUX starting. Environment=%s Watchlist=%s", cfg.raw.get("mode", {}).get("environment"), watchlist)

    while True:
        try:
            if kill_switch.is_triggered(kill_cfg.get("flag_file", "KILL_SWITCH")):
                kill_switch.flatten_all(client)
                open_positions.clear()
                log.critical("Kill switch engaged. Bot halted.")
                break

            now = now_et()
            today = now.date()
            if today != current_day:
                session_state.reset_for_new_session()
                pivots_by_symbol.clear()
                current_day = today
                log.info("New session day: %s", today)

            if not client.is_market_open():
                time.sleep(loop_seconds)
                continue

            vix_level = get_vix_level(
                vix_cfg.get("symbol", "^VIX"), int(vix_cfg.get("refresh_seconds", 60))
            )
            vix_sizing = classify_vix(
                vix_level,
                full_size_max=float(vix_cfg.get("full_size_max", 25.0)),
                half_size_max=float(vix_cfg.get("half_size_max", 30.0)),
                half_size_min_dte=int(vix_cfg.get("half_size_min_dte", 1)),
                half_size_max_dte=int(vix_cfg.get("half_size_max_dte", 3)),
                extreme_min_dte=int(vix_cfg.get("extreme_min_dte", 1)),
                half_size_multiplier=float(sizing_cfg.get("half_size_multiplier", 0.5)),
            )

            equity = client.get_account_equity()

            for symbol in watchlist:
                try:
                    _process_symbol(
                        symbol=symbol,
                        client=client,
                        loggers=loggers,
                        now=now,
                        today=today,
                        bar_timeframe=bar_timeframe,
                        sig_cfg=sig_cfg,
                        vix_sizing=vix_sizing,
                        btc_cfg=btc_cfg,
                        dte_cfg=dte_cfg,
                        contracts_cfg=contracts_cfg,
                        sizing_cfg=sizing_cfg,
                        exits_cfg=exits_cfg,
                        orders_cfg=orders_cfg,
                        no_entries_before=no_entries_before,
                        force_close_by=force_close_by,
                        session_state=session_state,
                        open_positions=open_positions,
                        pivots_by_symbol=pivots_by_symbol,
                        equity=equity,
                    )
                except Exception:
                    log.exception("Error processing %s", symbol)

            time.sleep(loop_seconds)
        except KeyboardInterrupt:
            log.info("Interrupted by user, shutting down.")
            break
        except Exception:
            log.exception("Unhandled error in main loop, continuing after a pause")
            time.sleep(loop_seconds)


def _process_symbol(
    symbol, client, loggers, now, today, bar_timeframe, sig_cfg, vix_sizing, btc_cfg,
    dte_cfg, contracts_cfg, sizing_cfg, exits_cfg, orders_cfg, no_entries_before,
    force_close_by, session_state: SessionRiskState, open_positions: dict, pivots_by_symbol: dict,
    equity: float,
) -> None:
    start = datetime.combine(today, datetime.min.time())
    bars = client.get_stock_bars(symbol, bar_timeframe, start=start, end=now)
    if bars.empty or len(bars) < 2:
        return

    pos = open_positions.get(symbol)
    if pos is not None:
        mid = _get_option_mid(client, symbol, pos.expiration, pos.option_symbol)
        if mid is not None:
            force_close = pos.expiration == today and is_force_close_time(now, force_close_by)
            exit_reason = should_exit_position(
                entry_price=pos.entry_price,
                current_price=mid,
                entry_time=pos.entry_time,
                now=now,
                stop_loss_pct=float(exits_cfg.get("stop_loss_pct", 40.0)),
                profit_target_pct=float(exits_cfg.get("profit_target_pct", 60.0)),
                time_stop_minutes=int(exits_cfg.get("time_stop_minutes", 120)),
            )
            if force_close and not exit_reason:
                exit_reason = "force_close_0dte"

            if exit_reason:
                result = submit_exit_limit_order(
                    client, pos.option_symbol, pos.contracts, mid,
                    float(orders_cfg.get("mid_offset_pct", 1.0)),
                    int(orders_cfg.get("cancel_unfilled_after_seconds", 20)),
                    int(orders_cfg.get("max_repricing_attempts", 3)),
                )
                loggers.orders.write_row({
                    "timestamp": now.isoformat(), "symbol": symbol, "option_symbol": pos.option_symbol,
                    "side": "sell", "qty": pos.contracts, "limit_price": mid, "order_id": result.order_id,
                    "filled": result.filled, "attempts": result.attempts, "reason": result.reason or "",
                })
                if result.qty > 0:
                    exit_price = result.avg_fill_price or mid
                    gross_pnl = (exit_price - pos.entry_price) * result.qty * 100
                    loggers.fills.write_row({
                        "timestamp": now.isoformat(), "symbol": symbol, "option_symbol": pos.option_symbol,
                        "side": "sell", "qty": result.qty, "avg_fill_price": exit_price, "order_id": result.order_id,
                    })
                    session_state.record_realized_pnl(gross_pnl)
                    loggers.pnl.write_row({
                        "timestamp": now.isoformat(), "symbol": symbol, "option_symbol": pos.option_symbol,
                        "direction": pos.direction, "entry_price": pos.entry_price, "exit_price": exit_price,
                        "contracts": result.qty, "gross_pnl": gross_pnl, "net_pnl": gross_pnl,
                        "exit_reason": exit_reason, "session_realized_pnl": session_state.realized_pnl,
                        "trades_today": session_state.trades_taken,
                    })
                    log.info("Closed %s %s qty=%d pnl=%.2f reason=%s", symbol, pos.option_symbol, result.qty, gross_pnl, exit_reason)
                    if result.filled:
                        del open_positions[symbol]
                    else:
                        # Only part of the position closed before the cancel/
                        # reprice budget ran out; keep the remainder open so
                        # it's picked up again next loop iteration.
                        pos.contracts -= result.qty
                        log.warning(
                            "Exit for %s only partially filled (%d closed, %d remain): %s",
                            symbol, result.qty, pos.contracts, result.reason,
                        )
                else:
                    log.warning("Exit order for %s did not fill: %s", symbol, result.reason)
        return  # one position at a time per symbol; don't also evaluate a fresh entry this cycle

    can_trade, block_reason = session_state.can_open_new_trade()
    if not can_trade:
        return
    if not is_entry_allowed(now, no_entries_before) or is_force_close_time(now, force_close_by):
        return

    pivots_key = f"{symbol}:{today}"
    if pivots_key not in pivots_by_symbol:
        prior = client.get_prior_session_ohlc(symbol, as_of=today)
        pivots_by_symbol[pivots_key] = compute_floor_pivots(prior["high"], prior["low"], prior["close"])
    pivots = pivots_by_symbol[pivots_key]

    signal = evaluate_signal(
        symbol, bars, pivots,
        ema_fast_span=int(sig_cfg.get("ema_fast", 9)),
        ema_slow_span=int(sig_cfg.get("ema_slow", 21)),
        volume_lookback=int(sig_cfg.get("volume_avg_lookback", 20)),
        volume_multiplier=float(sig_cfg.get("volume_multiplier", 1.0)),
        pivot_reclaim_buffer_pct=float(sig_cfg.get("pivot_reclaim_buffer_pct", 0.05)),
    )
    if signal is None:
        return
    loggers.signals.write_row(signal.as_log_row())
    if not signal.direction:
        return

    gated = {s.upper() for s in btc_cfg.get("gated_symbols", ["MSTR", "COIN"])}
    if symbol.upper() in gated:
        btc_price = get_btc_price(client, btc_cfg.get("symbol", "BTC/USD"), int(btc_cfg.get("refresh_seconds", 60)))
        if not btc_gate_allows(symbol, btc_price, float(btc_cfg.get("min_price_usd", 71000)), tuple(gated)):
            log.info("BTC gate blocked %s long (BTC=%.0f)", symbol, btc_price)
            return

    min_dte = vix_sizing.min_dte
    max_dte = vix_sizing.max_dte
    contract = select_contract(
        client, symbol, signal.direction, signal.price,
        min_dte=min_dte, max_dte=max_dte, allow_0dte=vix_sizing.allow_0dte,
        min_open_interest=int(contracts_cfg.get("min_open_interest", 100)),
        max_bid_ask_spread_pct=float(contracts_cfg.get("max_bid_ask_spread_pct", 10.0)),
        max_bid_ask_spread_abs=float(contracts_cfg.get("max_bid_ask_spread_abs", 0.15)),
        fallback_max_dte=int(dte_cfg.get("fallback_max_dte", 2)),
        as_of=today,
    )
    if contract is None:
        log.info("No qualifying contract for %s %s", symbol, signal.direction)
        return

    contracts_n = calc_contracts(
        equity, float(sizing_cfg.get("risk_pct_per_trade", 1.0)), vix_sizing.size_multiplier,
        contract.mid, int(sizing_cfg.get("max_contracts_per_trade", 10)),
    )
    if contracts_n <= 0:
        log.info("Sizing produced 0 contracts for %s (equity too small for risk cap at this premium)", symbol)
        return

    result = submit_entry_limit_order(
        client, contract, contracts_n,
        float(orders_cfg.get("mid_offset_pct", 1.0)),
        int(orders_cfg.get("cancel_unfilled_after_seconds", 20)),
        int(orders_cfg.get("max_repricing_attempts", 3)),
    )
    loggers.orders.write_row({
        "timestamp": now.isoformat(), "symbol": symbol, "option_symbol": contract.symbol,
        "side": "buy", "qty": contracts_n, "limit_price": contract.mid, "order_id": result.order_id,
        "filled": result.filled, "attempts": result.attempts, "reason": result.reason or "",
    })
    if result.qty > 0:
        entry_price = result.avg_fill_price or contract.mid
        loggers.fills.write_row({
            "timestamp": now.isoformat(), "symbol": symbol, "option_symbol": contract.symbol,
            "side": "buy", "qty": result.qty, "avg_fill_price": entry_price, "order_id": result.order_id,
        })
        open_positions[symbol] = OpenPosition(
            symbol=symbol, option_symbol=contract.symbol, direction=signal.direction,
            contracts=result.qty, entry_price=entry_price, entry_time=now,
            expiration=contract.expiration_date, strike=contract.strike,
        )
        session_state.record_trade_opened()
        log.info("Opened %s %s x%d @ %.2f%s", symbol, contract.symbol, result.qty, entry_price,
                  "" if result.filled else f" (partial, wanted {contracts_n})")
    else:
        log.warning("Entry order for %s did not fill: %s", symbol, result.reason)


def parse_args():
    p = argparse.ArgumentParser(description="FLUX options trading bot")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(config_path=args.config)
    setup_console_logging(cfg.get("logging", "level", default="INFO"))
    run(cfg)


if __name__ == "__main__":
    main()
