"""Contract selection: nearest ATM or 1-strike-OTM, filtered by minimum open
interest and maximum bid-ask spread.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from alpaca.trading.enums import ContractType

from flux.data.alpaca_client import FluxAlpacaClient

log = logging.getLogger("flux.execution.options_selector")


@dataclass(frozen=True)
class SelectedContract:
    symbol: str
    underlying_symbol: str
    expiration_date: date
    strike: float
    option_type: str  # "call" | "put"
    bid: float
    ask: float
    mid: float
    open_interest: int
    dte: int


def _parse_oi(value: Optional[str]) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def select_contract(
    client: FluxAlpacaClient,
    underlying_symbol: str,
    direction: str,
    underlying_price: float,
    min_dte: int,
    max_dte: Optional[int],
    allow_0dte: bool,
    min_open_interest: int,
    max_bid_ask_spread_pct: float,
    max_bid_ask_spread_abs: float,
    fallback_max_dte: int,
    as_of: Optional[date] = None,
) -> Optional[SelectedContract]:
    """Pick the nearest-expiration, ATM-or-1-strike-OTM contract that passes
    the open interest and bid-ask spread filters. Returns None if nothing
    qualifies.
    """
    today = as_of or date.today()
    contract_type = ContractType.CALL if direction == "call" else ContractType.PUT

    effective_min_dte = 0 if (allow_0dte and min_dte == 0) else max(min_dte, 0 if allow_0dte else 1)
    effective_max_dte = max_dte if max_dte is not None else fallback_max_dte
    if not allow_0dte:
        effective_min_dte = max(effective_min_dte, 1)

    exp_gte = today + timedelta(days=effective_min_dte)
    exp_lte = today + timedelta(days=effective_max_dte)

    contracts = client.get_option_contracts(
        underlying_symbol,
        expiration_date_gte=exp_gte,
        expiration_date_lte=exp_lte,
        contract_type=contract_type,
    )
    contracts = [c for c in contracts if _parse_oi(c.open_interest) >= min_open_interest]
    if not contracts:
        log.info("No contracts for %s passed the open-interest filter", underlying_symbol)
        return None

    nearest_exp = min(c.expiration_date for c in contracts)
    same_exp = [c for c in contracts if c.expiration_date == nearest_exp]

    same_exp.sort(key=lambda c: c.strike_price)
    strikes = [c.strike_price for c in same_exp]
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))

    if direction == "call":
        otm_idx = atm_idx + 1 if strikes[atm_idx] <= underlying_price else atm_idx
        otm_idx = min(otm_idx, len(strikes) - 1)
    else:
        otm_idx = atm_idx - 1 if strikes[atm_idx] >= underlying_price else atm_idx
        otm_idx = max(otm_idx, 0)

    candidate_idxs = [atm_idx] if otm_idx == atm_idx else [atm_idx, otm_idx]
    snapshot = client.get_option_chain_snapshot(underlying_symbol, expiration_date=nearest_exp)

    for idx in candidate_idxs:
        contract = same_exp[idx]
        snap = snapshot.get(contract.symbol) if snapshot else None
        if not snap or not snap.latest_quote:
            continue
        bid = float(snap.latest_quote.bid_price or 0)
        ask = float(snap.latest_quote.ask_price or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_pct = (spread / mid * 100.0) if mid > 0 else float("inf")
        if spread_pct > max_bid_ask_spread_pct or spread > max_bid_ask_spread_abs:
            log.info(
                "%s spread filter failed: spread=%.2f (%.1f%%) > caps",
                contract.symbol, spread, spread_pct,
            )
            continue
        return SelectedContract(
            symbol=contract.symbol,
            underlying_symbol=underlying_symbol,
            expiration_date=contract.expiration_date,
            strike=contract.strike_price,
            option_type=direction,
            bid=bid,
            ask=ask,
            mid=mid,
            open_interest=_parse_oi(contract.open_interest),
            dte=(contract.expiration_date - today).days,
        )

    log.info("No candidate contract for %s passed the spread filter", underlying_symbol)
    return None
