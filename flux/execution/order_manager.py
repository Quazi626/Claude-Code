"""Limit-order execution: price at mid with a small offset, cancel if
unfilled after N seconds (optionally reprice and retry up to a max attempts),
plus exit-side (sell to close) order handling.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from alpaca.trading.enums import OrderSide, PositionIntent

from flux.data.alpaca_client import FluxAlpacaClient
from flux.execution.options_selector import SelectedContract

log = logging.getLogger("flux.execution.order_manager")

TERMINAL_FILLED = {"filled"}
TERMINAL_DEAD = {"canceled", "expired", "rejected", "suspended", "stopped"}


@dataclass
class OrderResult:
    filled: bool
    symbol: str
    qty: int
    avg_fill_price: Optional[float]
    order_id: Optional[str]
    attempts: int
    reason: Optional[str] = None


def _order_status(order) -> str:
    status = order.status
    return status.value if hasattr(status, "value") else str(status)


def _poll_until_terminal(client: FluxAlpacaClient, order_id: str, timeout_seconds: int, poll_interval: float = 1.0):
    deadline = time.time() + timeout_seconds
    order = client.get_order(order_id)
    while time.time() < deadline:
        status = _order_status(order)
        if status in TERMINAL_FILLED or status in TERMINAL_DEAD:
            return order
        time.sleep(poll_interval)
        order = client.get_order(order_id)
    return order


def submit_entry_limit_order(
    client: FluxAlpacaClient,
    contract: SelectedContract,
    qty: int,
    mid_offset_pct: float,
    cancel_after_seconds: int,
    max_repricing_attempts: int,
) -> OrderResult:
    """Buy-to-open at mid + a small offset (pay up slightly to improve fill
    odds while still capping the price), cancelling and retrying with a
    slightly wider offset if unfilled after `cancel_after_seconds`.
    """
    return _submit_with_retries(
        client=client,
        symbol=contract.symbol,
        qty=qty,
        side=OrderSide.BUY,
        position_intent=PositionIntent.BUY_TO_OPEN,
        mid=contract.mid,
        mid_offset_pct=mid_offset_pct,
        cancel_after_seconds=cancel_after_seconds,
        max_repricing_attempts=max_repricing_attempts,
    )


def submit_exit_limit_order(
    client: FluxAlpacaClient,
    symbol: str,
    qty: int,
    mid: float,
    mid_offset_pct: float,
    cancel_after_seconds: int,
    max_repricing_attempts: int,
) -> OrderResult:
    """Sell-to-close at mid - a small offset (accept slightly less to
    improve fill odds), same cancel/retry behavior as entries.
    """
    return _submit_with_retries(
        client=client,
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        position_intent=PositionIntent.SELL_TO_CLOSE,
        mid=mid,
        mid_offset_pct=mid_offset_pct,
        cancel_after_seconds=cancel_after_seconds,
        max_repricing_attempts=max_repricing_attempts,
    )


def _submit_with_retries(
    client: FluxAlpacaClient,
    symbol: str,
    qty: int,
    side: OrderSide,
    position_intent: PositionIntent,
    mid: float,
    mid_offset_pct: float,
    cancel_after_seconds: int,
    max_repricing_attempts: int,
) -> OrderResult:
    attempts = 0
    current_offset_pct = mid_offset_pct
    last_order_id = None
    remaining_qty = qty
    filled_qty_total = 0
    filled_notional_total = 0.0  # sum of fill_price * fill_qty, for a weighted avg

    def _record_fill(order, limit_price: float) -> int:
        nonlocal filled_qty_total, filled_notional_total
        fq = int(float(order.filled_qty)) if order.filled_qty else 0
        fq = min(fq, remaining_qty)
        if fq <= 0:
            return 0
        price = float(order.filled_avg_price) if order.filled_avg_price else limit_price
        filled_qty_total += fq
        filled_notional_total += price * fq
        return fq

    while attempts < max(1, max_repricing_attempts) and remaining_qty > 0:
        attempts += 1
        # BUY pays up (higher limit improves fill odds); SELL accepts less.
        sign = 1 if side == OrderSide.BUY else -1
        limit_price = max(0.01, mid * (1 + sign * current_offset_pct / 100.0))

        client_order_id = f"flux-{uuid.uuid4().hex[:16]}"
        order = client.submit_limit_order(
            option_symbol=symbol,
            qty=remaining_qty,
            side=side,
            limit_price=limit_price,
            position_intent=position_intent,
            client_order_id=client_order_id,
        )
        last_order_id = order.id
        log.info(
            "Submitted %s order for %s qty=%d limit=%.2f (attempt %d/%d)",
            side.value, symbol, remaining_qty, limit_price, attempts, max_repricing_attempts,
        )

        final = _poll_until_terminal(client, str(order.id), cancel_after_seconds)
        status = _order_status(final)

        if status not in TERMINAL_FILLED and status not in TERMINAL_DEAD:
            # Still open (new/partially_filled) after the wait window: cancel
            # before retrying, so we never have two live orders for the same
            # intent stacking fills on top of each other.
            try:
                client.cancel_order(str(final.id))
            except Exception:
                log.warning("Cancel failed for order %s (may have just filled)", final.id, exc_info=True)
            final = client.get_order(str(final.id))
            status = _order_status(final)

        fq = _record_fill(final, limit_price)
        remaining_qty -= fq

        if status in TERMINAL_FILLED or remaining_qty <= 0:
            avg_price = filled_notional_total / filled_qty_total if filled_qty_total else None
            return OrderResult(
                filled=filled_qty_total >= qty, symbol=symbol, qty=filled_qty_total,
                avg_fill_price=avg_price, order_id=str(final.id), attempts=attempts,
                reason=None if filled_qty_total >= qty else "partially_filled",
            )

        log.info(
            "Order %s filled %d/%d after %ds (status=%s), repricing remainder",
            final.id, fq, qty, cancel_after_seconds, status,
        )
        current_offset_pct *= 1.5  # widen the offset each retry to improve fill odds

    avg_price = filled_notional_total / filled_qty_total if filled_qty_total else None
    return OrderResult(
        filled=False, symbol=symbol, qty=filled_qty_total, avg_fill_price=avg_price,
        order_id=last_order_id, attempts=attempts,
        reason="partially_filled_after_max_attempts" if filled_qty_total else "unfilled_after_max_attempts",
    )
