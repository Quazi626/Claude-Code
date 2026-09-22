"""CLI entrypoint for backtests.

Usage:
    python -m flux.backtest.run --start 2025-06-01 --end 2025-08-01 [--symbols SPY,QQQ] [--config config.yaml]

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY in .env (paper keys are fine — this
only reads historical market data, it never places orders).
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

from flux.backtest.data_loader import (
    load_btc_daily,
    load_daily_ohlc,
    load_intraday_bars,
    load_vix_daily,
)
from flux.backtest.engine import Trade, run_backtest
from flux.backtest.metrics import compute_metrics
from flux.config import load_config
from flux.data.alpaca_client import FluxAlpacaClient

log = logging.getLogger("flux.backtest.run")


def parse_args():
    p = argparse.ArgumentParser(description="FLUX backtester")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbols", default=None, help="Comma-separated; defaults to config.yaml watchlist")
    p.add_argument("--config", default=None)
    p.add_argument("--starting-equity", type=float, default=25000.0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(config_path=args.config)
    client = FluxAlpacaClient.from_creds(cfg.creds)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    symbols = args.symbols.split(",") if args.symbols else cfg.get("watchlist", default=[])

    vix_daily = load_vix_daily(start, end)

    all_trades: list[Trade] = []
    per_symbol_results = {}
    for symbol in symbols:
        symbol = symbol.strip()
        log.info("Backtesting %s from %s to %s", symbol, start, end)
        bars = load_intraday_bars(client, symbol, start, end, timeframe=cfg.get("signals", "bar_timeframes", default=["5Min"])[0])
        if bars.empty:
            log.warning("No bars for %s, skipping", symbol)
            continue
        daily_ohlc = load_daily_ohlc(client, symbol, start, end)

        btc_daily = None
        gated = {s.upper() for s in cfg.get("btc_gate", "gated_symbols", default=["MSTR", "COIN"])}
        if symbol.upper() in gated:
            btc_daily = load_btc_daily(client, start, end)

        trades = run_backtest(
            symbol, bars, daily_ohlc, vix_daily, cfg,
            btc_daily=btc_daily, starting_equity=args.starting_equity,
        )
        all_trades.extend(trades)
        per_symbol_results[symbol] = compute_metrics(trades, args.starting_equity)

    print("\n" + "=" * 70)
    print("PER-SYMBOL RESULTS")
    print("=" * 70)
    for symbol, metrics in per_symbol_results.items():
        print(f"\n{symbol}:")
        print("  " + metrics.summary().replace("\n", "\n  "))

    print("\n" + "=" * 70)
    print("PORTFOLIO RESULTS")
    print("=" * 70)
    overall = compute_metrics(all_trades, args.starting_equity)
    print(overall.summary())


if __name__ == "__main__":
    main()
