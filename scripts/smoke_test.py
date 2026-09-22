#!/usr/bin/env python3
"""Connectivity smoke test — verifies your .env/config.yaml setup can reach
every data source FLUX depends on, WITHOUT placing any orders.

Run this first, before `python -m flux.main`, to sanity-check API keys and
account permissions in under a minute:

    python scripts/smoke_test.py

Each check is independent and reports pass/fail on its own; one failure
doesn't stop the rest from running, so you get a full picture in one pass.
Exit code is 0 only if every check passed.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flux.config import load_config
from flux.data.alpaca_client import FluxAlpacaClient
from flux.data.btc import get_btc_price
from flux.data.vix import get_vix_level

CHECK_SYMBOL = "SPY"

results: list[tuple[str, bool, str]] = []
cfg = None
client = None
_sample_expiration = None


def check(name: str):
    """Decorator-ish helper: runs `fn()`, records pass/fail + detail message."""
    def wrapper(fn):
        print(f"→ {name} ...", end=" ", flush=True)
        try:
            detail = fn()
            print("OK")
            if detail:
                print(f"    {detail}")
            results.append((name, True, detail or ""))
        except Exception as e:
            print("FAILED")
            print(f"    {type(e).__name__}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
            results.append((name, False, str(e)))
    return wrapper


def main() -> int:
    print("=" * 70)
    print("FLUX connectivity smoke test — no orders will be placed")
    print("=" * 70)

    print("\n[1/8] Config & credentials")

    @check("Load config.yaml + .env")
    def _load_cfg():
        global cfg
        cfg = load_config()
        env = cfg.raw.get("mode", {}).get("environment", "paper")
        base = cfg.creds.base_url
        if not cfg.creds.api_key or not cfg.creds.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are empty — check your .env file"
            )
        return f"environment={env}  base_url={base}"

    if not results or not results[-1][1]:
        print("\nCannot continue without a loaded config. Fix the above and re-run.")
        return 1

    print("\n[2/8] Alpaca trading client + account")

    @check("Build Alpaca clients")
    def _build_client():
        global client
        client = FluxAlpacaClient.from_creds(cfg.creds)
        return "trading / stock_data / option_data / crypto_data clients constructed"

    @check("Fetch account (auth check)")
    def _account():
        account = client.trading.get_account()
        equity = float(account.equity)
        status = account.status.value if hasattr(account.status, "value") else account.status
        options_level = getattr(account, "options_trading_level", None)
        detail = f"equity=${equity:,.2f}  status={status}"
        if options_level is not None:
            detail += f"  options_trading_level={options_level}"
        return detail

    print("\n[3/8] Market clock")

    @check("Fetch market clock")
    def _clock():
        clock = client.trading.get_clock()
        state = "OPEN" if clock.is_open else "CLOSED"
        return f"market is {state} (server time {clock.timestamp})"

    print("\n[4/8] Equities market data")

    @check(f"Fetch recent daily bars for {CHECK_SYMBOL}")
    def _bars():
        end = datetime.utcnow()
        start = end - timedelta(days=10)
        df = client.get_stock_bars(CHECK_SYMBOL, "1Day", start=start, end=end)
        if df.empty:
            raise RuntimeError("no bars returned — check market data subscription/entitlement")
        last_close = float(df["close"].iloc[-1])
        return f"{len(df)} daily bars, last close=${last_close:.2f}"

    print("\n[5/8] Options data (contracts + chain snapshot)")

    @check(f"Fetch option contracts for {CHECK_SYMBOL}")
    def _contracts():
        from datetime import date
        contracts = client.get_option_contracts(
            CHECK_SYMBOL,
            expiration_date_gte=date.today(),
            expiration_date_lte=date.today() + timedelta(days=14),
        )
        if not contracts:
            raise RuntimeError(
                "no option contracts returned — options trading may not be enabled "
                "on this account, or the symbol has no near-term listings"
            )
        global _sample_expiration
        _sample_expiration = min(c.expiration_date for c in contracts)
        return f"{len(contracts)} contracts found, nearest expiration={_sample_expiration}"

    @check(f"Fetch option chain snapshot for {CHECK_SYMBOL}")
    def _snapshot():
        if _sample_expiration is None:
            raise RuntimeError("skipped — the option contracts fetch above failed first")
        snapshot = client.get_option_chain_snapshot(CHECK_SYMBOL, expiration_date=_sample_expiration)
        if not snapshot:
            raise RuntimeError("empty snapshot — no quotes available for this expiration right now")
        quoted = sum(1 for s in snapshot.values() if s.latest_quote and s.latest_quote.bid_price)
        return f"{len(snapshot)} contracts in chain, {quoted} with a live bid"

    print("\n[6/8] Crypto market data (BTC gate)")

    @check("Fetch latest BTC/USD price")
    def _btc():
        price = get_btc_price(client, "BTC/USD", refresh_seconds=0)
        return f"BTC/USD = ${price:,.2f}"

    print("\n[7/8] VIX (yfinance, independent of Alpaca)")

    @check("Fetch VIX level")
    def _vix():
        level = get_vix_level("^VIX", refresh_seconds=0)
        return f"VIX = {level:.2f}"

    print("\n[8/8] Kill switch flag path")

    @check("Kill switch flag file is NOT currently set")
    def _kill_switch():
        from pathlib import Path
        flag = cfg.get("kill_switch", "flag_file", default="KILL_SWITCH")
        if Path(flag).exists():
            raise RuntimeError(
                f"'{flag}' exists in the repo root — the bot would refuse to trade "
                f"and flatten positions on startup. Delete it if that wasn't intended."
            )
        return f"'{flag}' not present (as expected)"

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 70)
    if passed < total:
        print("\nFailed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        print("\nFix these before running `python -m flux.main`.")
        return 1

    print("\nAll checks passed. You're good to run `python -m flux.main` in paper mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
