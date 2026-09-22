# FLUX — Single-Leg Options Trading Bot

A Python bot that buys single-leg calls/puts on US equity options (no
spreads, no naked selling) against Alpaca's API, using the "FLUX" intraday
framework: daily floor pivots + VWAP + 9/21 EMA cross + volume confirmation.

**Paper trading is the default and the safe path.** Live trading requires an
explicit config flag *and* a typed confirmation at startup — see
[Live trading](#live-trading-two-factor-gate) below.

## How FLUX decides a trade

1. **Floor pivots** (P, R1–R3, S1–S3) computed from the prior session's daily
   OHLC.
2. On 5m/15m bars: price vs session VWAP, 9-EMA vs 21-EMA, and current volume
   vs a 20-bar average.
3. **Long call**: price above VWAP + 9-EMA > 21-EMA + volume confirmation +
   price just reclaimed a pivot level from below.
   **Long put**: the mirror image (below VWAP, 9-EMA < 21-EMA, breakdown
   through a pivot level from above).
4. No new entries before 9:45 AM ET. All 0DTE positions are force-closed by
   3:15 PM ET.
5. **VIX sizing**: VIX < 25 → full size, 0DTE allowed. 25–30 → half size,
   1–3 DTE minimum (no 0DTE). ≥ 30 → half size, 0DTE blocked entirely.
6. **BTC gate**: MSTR/COIN longs are blocked while BTC/USD is below a
   configured floor (default $71,000).
7. Max 3–5 trades per session (configurable); the bot stops trading for the
   day the moment either the daily profit target or the daily max loss is
   hit (a one-way stop — it doesn't resume even if P&L drifts back).
8. Contracts are the nearest ATM or 1-strike-OTM, filtered by minimum open
   interest and maximum bid-ask spread.
9. Exits: configurable % stop-loss, % profit target, and a time stop, plus
   the 3:15 PM force-close for 0DTE.

## Project layout

```
flux/
  config.py              # config.yaml + .env loading, live-trading gate
  main.py                 # live/paper trading loop entrypoint
  logging_utils.py        # CSV + console logging
  data/
    alpaca_client.py       # Alpaca trading/data client wrappers
    pivots.py               # floor pivot math
    vix.py                   # VIX level via yfinance
    btc.py                    # BTC/USD price via Alpaca crypto data
  signals/
    indicators.py           # VWAP, EMA, rolling volume average
    flux.py                   # FLUX entry signal logic
  risk/
    rules.py                 # time gates, VIX sizing, BTC gate, session
                               # limits, daily stop, position sizing, exits
  execution/
    options_selector.py      # ATM/OTM contract selection + OI/spread filters
    order_manager.py          # limit orders at mid+offset, cancel/reprice
    kill_switch.py             # file-flag kill switch
  backtest/
    black_scholes.py          # option price simulation (no paid chain data)
    engine.py                   # historical bar replay through FLUX+risk
    metrics.py                   # win rate, avg win/loss, max drawdown
    data_loader.py                # historical bars/VIX/BTC assembly
    run.py                         # backtest CLI
tests/
  test_risk_rules.py         # time gates, VIX sizing, BTC gate, trade
                               # limits, daily stop, sizing, exit rules
  test_pivots.py
config.yaml                  # strategy/risk parameters (no secrets)
.env.example                 # copy to .env and fill in API keys
```

## Setup

1. **Python 3.11+** and a virtualenv:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Alpaca account**: create a free paper account at alpaca.markets, enable
   options trading (paper accounts get options access automatically), and
   generate a paper API key pair.

3. **Copy `.env.example` to `.env`** and fill in your paper keys:
   ```bash
   cp .env.example .env
   # edit .env: ALPACA_API_KEY, ALPACA_SECRET_KEY
   # ALPACA_BASE_URL should stay https://paper-api.alpaca.markets for paper mode
   ```
   `.env` is git-ignored — never commit real keys.

4. **Review `config.yaml`**: watchlist, entry/force-close times, VIX
   thresholds, BTC gate price, sizing, session limits, exit percentages,
   order offsets, and the kill-switch flag file path.

5. **Run the unit tests**:
   ```bash
   pytest tests/ -v
   ```

6. **Run the connectivity smoke test** — checks auth, account access, market
   clock, equity/options/crypto data, and VIX, without placing any orders:
   ```bash
   python scripts/smoke_test.py
   ```
   Fix anything it reports before moving on; it's the fastest way to catch a
   bad key, a paper account without options approval, or a data
   entitlement gap before the bot ever tries to trade.

7. **Run in paper mode**:
   ```bash
   python -m flux.main
   ```
   Logs stream to the console and to `logs/{signals,orders,fills,pnl}.csv`.

## Live trading (two-factor gate)

Live trading is deliberately hard to reach by accident:

1. Set `mode.environment: live` in `config.yaml`.
2. Set `ALPACA_BASE_URL` to the **live** endpoint
   (`https://api.alpaca.markets`) and put live API keys in `.env`.
3. Set `LIVE_TRADING_CONFIRM=YES` in `.env`.
4. At startup, type `I UNDERSTAND THE RISK` exactly when prompted.

Missing any one of these three raises an error and the bot refuses to
start in live mode. Paper mode needs none of this.

## Kill switch

Create the flag file named in `kill_switch.flag_file` (default
`KILL_SWITCH`, at the repo root) at any time:

```bash
touch KILL_SWITCH
```

The bot checks for it every loop iteration; when present, it cancels all
open orders, flattens every position, and halts. Delete the file before
restarting the bot. This is a file flag rather than a keypress listener so
it works identically whether the bot is running in a foreground terminal or
headless/backgrounded.

## Backtesting

Historical options chains aren't freely available, so the backtester
replays historical **underlying** bars from Alpaca through the same
FLUX/risk logic the live bot uses, and simulates option premiums with
Black-Scholes (flat IV assumption in `config.yaml: backtest.iv_assumption`)
plus configurable fees and slippage. Treat results as directional, not a
precise historical P&L — a real historical options chain (e.g. a paid
provider) would be needed for that.

```bash
python -m flux.backtest.run --start 2025-06-01 --end 2025-08-01 --symbols SPY,QQQ
```

Reports win rate, average win/loss, total P&L, max drawdown, and profit
factor, per symbol and for the combined portfolio of trades.

## Data sources

| Data | Source | Why |
|---|---|---|
| Equity bars, options chain/contracts, orders | Alpaca | Primary broker; paper by default |
| VIX index level | yfinance (`^VIX`) | Alpaca doesn't carry the CBOE VIX index |
| BTC/USD spot | Alpaca crypto market data | Same account, already authenticated |

## Safety notes

- **Single-leg long options only.** The order manager only ever submits
  buy-to-open / sell-to-close limit orders for one contract at a time —
  there is no spread or short-option code path.
- All order flow is limit orders priced at mid ± a configurable offset,
  with an unfilled-order cancel timeout and a capped number of repricing
  attempts.
- Per-trade sizing is capped as a percentage of account equity, further
  scaled down by the VIX regime.
- This is trading software; past backtest performance does not guarantee
  future results. Start in paper mode and validate behavior for a while
  before ever considering live trading.
